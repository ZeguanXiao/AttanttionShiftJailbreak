import os
from copy import deepcopy
import argparse
import rootutils
from time import sleep
from datetime import datetime

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.prompts import get_attacker_system_prompt, process_target_response, get_init_msg
from src.prompts import get_objective_string, get_start_string, get_rules, get_examples, get_examples_and_scores
from src.loggers import WandBLogger
from src.judges import load_judge
from src.conversers import load_attack_and_target_models
from src.common import conv_template, seed_everything
from src.data import load_data


def main(args):
    seed_everything(args.seed)
    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")
    starttime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_dir = os.path.join(root_path, "results", args.task_name, starttime)
    os.makedirs(run_dir)

    dataset = load_data(args.dataset, args.dataset_size)

    # Initialize models
    attackLM, targetLM = load_attack_and_target_models(args)
    judgeLM = load_judge(args)

    for global_iteration in range(args.n_global_iterations):
        if global_iteration == 0:
            # Set attacker system prompt
            objective_string, instruction_placeholder = get_objective_string()
            start_string = get_start_string()
            rules = get_rules(start_string, objective_string, instruction_placeholder)
            examples = get_examples(start_string, objective_string, args.example_group)
            system_prompt = get_attacker_system_prompt(rules, examples)

            logger = WandBLogger(args, system_prompt, save_dir=run_dir)
            print("*" * 30 + "Arguments" + "*" * 30)
            print(vars(args))

            print("*" * 10 + f"Saving results to {run_dir}." + "*" * 10)

            print(f"""\n{'=' * 36}\n Global Iteration: {global_iteration + 1}\n{'=' * 36}\n""")
            print("*" * 30 + "ATTACKER SYSTEM PROMPT" + "*" * 30)
            print(system_prompt)
            print("*" * 100)

            # Initialize conversations
            init_msg = get_init_msg(start_string)
            print(f"Init message: {init_msg}")
        else:
            current_top_attacks = deepcopy(logger.top_attacks)
            if args.add_init_examples:
                init_prompts = get_examples_and_scores(args.example_group,
                                                       args.target_model,
                                                       start_string,
                                                       objective_string)
                current_top_attacks += init_prompts
                current_top_attacks = sorted(current_top_attacks, key=lambda x: x['score'], reverse=True)

            top_attacks = []
            for attack in current_top_attacks:
                # Skip prompt that may too similar to previous prompts
                if any([(top_a["global_iteration"] == attack["global_iteration"]) and (
                        top_a["stream"] == attack["stream"]) for top_a in top_attacks]):
                    continue
                top_attacks.append(attack)
            top_attacks = top_attacks[:3]
            print("*" * 10 + f"Top {len(top_attacks)} attacks from previous global iteration" + "*" * 10)
            for attack in top_attacks:
                print(f"Score: {attack['score']} prompt: {attack['prompt']}")
                examples = f"""
EXAMPLES:
Here are some examples of adversarial prompt templates:
1) {top_attacks[0]["prompt"]}
2) {top_attacks[1]["prompt"]}
3) {top_attacks[2]["prompt"]}"""

            system_prompt = get_attacker_system_prompt(args.prompt_type, rules, examples)

            print(f"""\n{'=' * 36}\n Global Iteration: {global_iteration + 1}\n{'=' * 36}\n""")
            print("*" * 30 + "ATTACKER SYSTEM PROMPT" + "*" * 30)
            print(system_prompt)
            print("*" * 100)

        processed_response_list = [init_msg for _ in range(args.n_streams)]
        convs_list = [conv_template(attackLM.template) for _ in range(args.n_streams)]
        for conv in convs_list:
            conv.set_system_message(system_prompt)

        # Begin train
        for iteration in range(1, args.n_iterations + 1):
            print(f"""\n{'=' * 36}\nLocal Iteration: {iteration}\n{'=' * 36}\n""")
            if iteration > 1:
                processed_response_list = [process_target_response(score) for score in judge_scores]

            # Get adversarial prompts and improvement
            extracted_attack_list = attackLM.get_attack(convs_list,
                                                        processed_response_list,
                                                        args.attack_batch_size,
                                                        objective_string,
                                                        start_string)
            print("Finished getting adversarial prompts.")

            # Extract prompts and improvements
            adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
            improv_list = [attack["improvement"] for attack in extracted_attack_list]

            # Get target responses
            target_response_list = targetLM.get_response(adv_prompt_list,
                                                         dataset,
                                                         args.target_batch_size,
                                                         instruction_placeholder,
                                                         args.defense,
                                                         args.gen_num)
            print("Finished getting target responses.")
            judge_scores, judge_predictions, error_examples = judgeLM.score(target_response_list)
            print("Finished getting judge scores.")

            # Print prompts, responses, and scores
            for i, (prompt, improv, score) in enumerate(
                    zip(adv_prompt_list, improv_list, judge_scores)):
                print(
                    f"{i + 1}/{args.n_streams}\n\n[IMPROVEMENT]:\n{improv} \n\n[PROMPT]:\n{prompt} \n\n[SCORE]:\n{score}\n\n")

            # WandB log values
            logger.log(global_iteration,
                       iteration,
                       adv_prompt_list,
                       target_response_list,
                       judge_scores,
                       judge_predictions)

            # Truncate conversation to avoid context length issues
            for i, conv in enumerate(convs_list):
                conv.messages = conv.messages[-2 * (args.keep_last_n):]

            # Early stopping criterion
            if any([score == 100 for score in judge_scores]):
                print("Jailbreak all samples. Exiting.")
                break

            # Sleep
            sleep(args.sleeptime)

    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ########### Attack model parameters ##########
    parser.add_argument(
        "--attack-model",
        default="vicuna-13b",
        help="Name of attacking model.",
        choices=["vicuna-7b", "vicuna-13b", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4"]
    )
    parser.add_argument(
        "--attack-max-n-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens for the attacker."
    )
    parser.add_argument(
        "--max-n-attack-attempts",
        type=int,
        default=10,
        help="Maximum number of attack generation attempts, in case of generation errors."
    )
    parser.add_argument(
        "--attack-temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling."
    )
    parser.add_argument(
        "--attack-top-p",
        type=float,
        default=0.9,
        help="Top p for sampling."
    )
    parser.add_argument(
        "--attack-device",
        default="cuda:0",
        help="Device to use for attack model.",
    )
    parser.add_argument(
        "--attack-batch-size",
        type=int,
        default=5,
        help="Batch size for attack model."
    )
    ##################################################

    ########### Target model parameters ##########
    parser.add_argument(
        "--target-model",
        default="gpt-3.5-turbo-0613",
        help="Name of target model.",
        choices=["vicuna-7b", "vicuna-13b", "llama-2", "llama-2-sys", "llama-3", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4"]
    )
    parser.add_argument(
        "--target-max-n-tokens",
        type=int,
        default=2048,
        help="Maximum number of generated tokens for the target."
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling."
    )
    parser.add_argument(
        "--target-top-p",
        type=float,
        default=1.0,
        help="Top p for sampling."
    )
    parser.add_argument(
        "--target-device",
        default="cuda:0",
        help="Device to use for target model.",
    )
    parser.add_argument(
        "--gen_num",
        type=int,
        default=1,
        help="Number of generations per prompt."
    )
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=8,
        help="Batch size for target model."
    )
    ##################################################

    ############ Judge model parameters ##########
    parser.add_argument(
        "--judge-model",
        default="deberta",
        help="Name of judge model.",
        choices=["deberta",  "no-judge"]
    )
    parser.add_argument(
        "--judge-max-n-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens for the judge."
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0,
        help="Temperature to use for judge."
    )
    parser.add_argument(
        "--judge-model-device",
        default="cuda:0",
        help="Device to use for judge model.",
    )
    parser.add_argument(
        "--judge-model-batch-size",
        type=int,
        default=16,
        help="Batch size for judge model."
    )
    ##################################################

    ########### Dataset parameters ##########
    parser.add_argument(
        "--dataset",
        default="advbench_behaviors_custom",
        help="Name of dataset to use.",
        choices=["advbench_behaviors_custom"]
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=50,
        help="Number of samples to use from the dataset."
    )
    ##################################################

    ########### PAIR parameters ##########
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Name of wandb run.",
    )
    parser.add_argument(
        "--n-streams",
        type=int,
        default=10,
        help="Number of concurrent jailbreak conversations. If this is too large, then there may be out of memory errors."
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=3,
        help="Number of responses to save in conversation history of attack model. If this is too large, then it may exceed the context window of the model."
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=5,
        help="Number of local iterations to run the attack."
    )
    parser.add_argument(
        "--n-global-iterations",
        type=int,
        default=1,
        help="Number of global iterations to run the attack."
    )
    parser.add_argument(
        "--top-n-attack",
        type=int,
        default=5,
        help="Number of top attacks to check."
    )
    parser.add_argument(
        "--sleeptime",
        type=int,
        default=0,
        help="Sleep time between iterations."
    )
    parser.add_argument(
        "--example-group",
        type=int,
        default=1,
        help="Which group of prompts to use as examples in meta prompt.",
    )
    parser.add_argument(
        "--low-gpu-mem",
        action="store_true",
        help="Whether to use low GPU memory mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--add-init-examples",
        action="store_true",
        help="Whether to add examples in the initial meta prompt.",
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="none",
        choices=['none', 'sr', 'ic'],
        help="LLM defense: None, Self-Reminder, In-Context"
    )
    ######
    ##################################################

    args = parser.parse_args()

    main(args)
