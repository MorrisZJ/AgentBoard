import os
import json
from types import SimpleNamespace

from dotenv import load_dotenv

from agentboard.eval_main import load_config, check_log_paths_are_ready
from tasks import load_task
from llm import load_llm
from utils.logging.logger import SummaryLogger
from utils.logging.agent_logger import AgentLogger


logger = AgentLogger(__name__)


def main():
    """
    Debug runner for the AlfWorld task.

    - Reuses the existing config / task / LLM loading logic.
    - Forces num_exam = 3 so that only three examples are evaluated.
    - Does not modify any existing functions or config files.
    """
    load_dotenv()

    # You can edit these three variables locally if needed
    cfg_path = "eval_configs/main_results_all_tasks.yaml"
    task_name = "alfworld"
    model_name = "gpt-3.5-turbo-0613"

    # Prepare synthetic args for load_config (mimic eval_main.parse_args result)
    args = SimpleNamespace(
        cfg_path=cfg_path,
        tasks=[task_name],
        model=model_name,
        wandb=False,
        log_path="./results/debug_alfworld",
        project_name="debug-alfworld",
        baseline_dir="./data/baseline_results",
        max_num_steps=30,
    )

    llm_config_all, agent_config, env_config, run_config = load_config(cfg_path, args)
    llm_config = llm_config_all[model_name]

    # Limit to 3 examples for fast debugging
    run_config["num_exam"] = 3

    # Disable wandb explicitly for debug
    run_config["wandb"] = False
    os.environ["WANDB_MODE"] = "disabled"
    logger.info("Wandb is disabled in debug_alfworld.")

    log_dir = run_config.get("log_path", "./results/debug_alfworld")
    baseline_path = run_config.get("baseline_dir", "./data/baseline_results")

    assert check_log_paths_are_ready(log_dir, baseline_path)

    agentboard = SummaryLogger(baseline_dir=baseline_path, log_path=log_dir)

    # Load LLM
    logger.info("Start loading language model (debug_alfworld)")
    llm = load_llm(llm_config["name"], llm_config)
    logger.info("Finished loading language model (debug_alfworld)")

    # Merge env-specific fields into agent config (same as eval_main)
    agent_task_config = agent_config.copy()
    for key in env_config[task_name]:
        if key in ["check_actions", "check_inventory", "init_prompt_path"]:
            agent_task_config[key] = env_config[task_name][key]

    # Load task and run evaluation
    logger.info(f"Start evaluating task {task_name} with num_exam=3 (debug)")
    task = load_task(task_name, run_config, llm_config, agent_task_config, env_config[task_name], llm=llm)

    success_rates, progress_rates, grounding_accs, score_state_records, \
        easy_sr, hard_sr, easy_pr, hard_pr = task.evaluate()

    success_rate = sum(success_rates) * 1.0 / len(success_rates)
    progress_rate = sum(progress_rates) * 1.0 / len(progress_rates)
    grounding_acc = sum(grounding_accs) * 1.0 / len(grounding_accs)

    logger.finish(
        f"[DEBUG] Task {task_name} | Success Rate: {success_rate} , Progress Rate: {progress_rate} , "
        f"Easy SR: {easy_sr}. Hard SR: {hard_sr}, Easy PR: {easy_pr}, Hard PR: {hard_pr}, "
        f"Grounding Accuracy: {grounding_acc}"
    )

    agentboard.log_run_result(
        task_name,
        success_rate,
        progress_rate,
        grounding_acc,
        hard_sr,
        easy_sr,
        hard_pr,
        easy_pr,
    )

    logger.info("Finish debug evaluation for alfworld (3 examples).")
    agentboard.log_summary()


if __name__ == "__main__":
    main()

