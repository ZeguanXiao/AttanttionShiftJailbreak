import os
import copy
import wandb
import pandas as pd


class WandBLogger:
    """WandB logger."""

    def __init__(self, args, system_prompt, save_dir):
        self.save_dir = save_dir
        config_dict = copy.deepcopy(vars(args))
        config_dict["system_prompt"] = system_prompt

        self.logger = wandb.init(
            name=args.task_name if args.task_name else "default",
            project="jailbreak-llms",
            config=config_dict,
        )
        wandb.run.log_code("./src")
        self.args = args
        self.table = pd.DataFrame()
        self.top_attacks = []

    def log(self, global_iteration: int, iteration: int, prompt_list: list, response_list: list, judge_scores: list, judge_predictions: list):
        # Log details to WandB
        df = pd.DataFrame(
            [output_dict for stream_output_dict_list in response_list for output_dict in stream_output_dict_list])
        df["prediction"] = [p for sublist in judge_predictions for p in sublist]
        df["global_iter"] = global_iteration
        df["iter"] = iteration
        self.table = pd.concat([self.table, df])

        for i, (score, prompt, sublist) in enumerate(zip(judge_scores, prompt_list, judge_predictions)):
            self.top_attacks.append({"score": score, "prompt":  prompt, "predictions": sublist,
                                     "iteration": iteration, "stream": i, "global_iteration": global_iteration})
            self.top_attacks = sorted(self.top_attacks, key=lambda x: x['score'], reverse=True)

        self.logger.log({"iteration": iteration, "top_1_score": self.top_attacks[0]["score"]})
        self.print_summary_stats(judge_scores)
        self.table.to_csv(os.path.join(self.save_dir, "results.csv"), index=False)
        pd.DataFrame(self.top_attacks).to_csv(os.path.join(self.save_dir, "top_attacks.csv"), index=False)

    def finish(self):
        self.print_final_summary_stats()
        self.logger.finish()

    def print_summary_stats(self, judge_scores):
        max_score_for_iter = max(judge_scores)

        print(f"{'=' * 14} SUMMARY STATISTICS {'=' * 14}")
        print(f"Max Score for the iteration: {max_score_for_iter}")

    def print_final_summary_stats(self):
        max_score = self.top_attacks[0]["score"]
        max_score_prompt = self.top_attacks[0]["prompt"]
        top_n_predictions = [attack["predictions"] for attack in self.top_attacks[:self.args.top_n_attack]]
        top_n_attack_results = [any(sample_predictions) for sample_predictions in zip(*top_n_predictions)]
        top_n_score = round(sum(top_n_attack_results) / len(top_n_attack_results) * 100, 2)

        self.logger.log({"max_score": max_score,
                         "top_n_score": top_n_score,
                         "top_n_attacks": wandb.Table(data=pd.DataFrame(self.top_attacks)),
                         "data": wandb.Table(data=self.table)})

        print(f"{'=' * 8} FINAL SUMMARY STATISTICS {'=' * 8}")
        print(f"Max Score: {max_score}")
        print(f"Max Score Attack Prompt: {max_score_prompt}")
        print(f"Top {self.args.top_n_attack} Attacks Score: {top_n_score}")
