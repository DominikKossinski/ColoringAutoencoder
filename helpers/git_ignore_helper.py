import os


def create_model_gitignore(model_path: str) -> None:
    gitignore_lines = [
        "val_models\n"
        "val_images\n"
    ]
    with open(os.path.join(model_path, ".gitignore"), "w") as file:
        file.writelines(gitignore_lines)
        file.close()
