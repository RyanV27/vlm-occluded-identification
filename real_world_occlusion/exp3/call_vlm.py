import json
import argparse

from pathlib import Path
from PIL import Image

# ----------------- Do this if running the script inside exp2/------------------ #
import sys
import os

# Adding parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
# ---------------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Prompts the VLM with the environment images to identify the objects in each scene.")

    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        choices=["Qwen2.5-VL", "InternVL3.5", "Molmo2", "Molmo", "FT-Qwen2.5-VL",
                  "Gemma3n", "Llama-3.2", "FT-Gemma3", "FT-Llama-3.2", "Gemma3",
                  "Qwen3-VL", "FT-Qwen3-VL", "COCO-FT-Llama-3.2", "COCO-FT-Gemma3", "COCO-FT-Qwen2.5-VL", 
                "COCO-FT-Qwen3-VL"],
        help="Name of the VLM to test."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./runs/",
        help=(
            "Path to output directory where the model outputs should be stored."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/drive/MyDrive/Thesis/exp3",
        help=(
            "Path to data directory where environment images and a JSON",
            "file where an object list for each environment is stored."
        )
    )
    parser.add_argument(
        "--model_cache_path",
        type=str,
        default="/scratch/rsvargh2/huggingface_models/",
        help=(
            "Path to data directory where models cache files are stored."
        )
    )

    args = parser.parse_args()

    # Setting path variables
    data_path = Path(args.data_path)
    metadata_file_path = data_path / "data.json"

    # Reading the metadata about the images
    with open(metadata_file_path, "r") as f_in:
        metadata = json.load(f_in)

    # Setting and creating a file to store model generations
    out_dir_path = Path(args.out_path)
    out_dir_path.mkdir(exist_ok=True)
    out_path = out_dir_path / f"{args.model_name}_generations.json"

    # Load existing generations if the output file already exists
    if out_path.exists():
        with open(out_path, "r") as f_in:
            overall_scenes_generations = json.load(f_in)
        print(f"Loaded {len(overall_scenes_generations)} existing generations from {out_path}")
        if len(overall_scenes_generations) == len(metadata):
            print(f"All generations are present for {args.model_name} model. Exiting code.")
            return
    else:
        overall_scenes_generations = {}
        # overall_scenes_generations = []

    # Loading the model to test
    if args.model_name == "Qwen2.5-VL":
        from models import Qwen25VL
        model = Qwen25VL(cache_dir=args.model_cache_path)
    elif args.model_name == "InternVL3.5":
        from models import InternVL35
        model = InternVL35(cache_dir=args.model_cache_path)
    elif args.model_name == "Molmo2":
        from models import Molmo2
        model = Molmo2(cache_dir=args.model_cache_path)
    elif args.model_name == "Molmo":
        from models import Molmo
        model = Molmo(cache_dir=args.model_cache_path)
    elif args.model_name == "Gemma3":
        from models import Gemma3
        model = Gemma3()
    elif args.model_name == "Llama-3.2":
        from models import Llama3_2
        model = Llama3_2()
    elif args.model_name == "Qwen3-VL":
        from models import Qwen3VL
        model = Qwen3VL()
        
    
    # Models finetuned on tabletop images 
    elif args.model_name == "FT-Qwen2.5-VL":
        from models import FTQwen25VL
        model = FTQwen25VL()
    elif args.model_name == "FT-Gemma3":
        from models import FTGemma3
        model = FTGemma3()
    # elif args.model_name == "FT-Gemma3n":
    #     from models import FTGemma3n
    #     model = FTGemma3n()
    elif args.model_name == "FT-Llama-3.2":
        from models import FTLlama3_2
        model = FTLlama3_2()
    elif args.model_name == "FT-Qwen3-VL":
        from models import FTQwen3VL
        model = FTQwen3VL()

    # Models finetuned on COCO images
    elif args.model_name == "COCO-FT-Qwen2.5-VL":
        from models import COCOFTQwen25VL
        model = COCOFTQwen25VL()
    elif args.model_name == "COCO-FT-Gemma3":
        from models import COCOFTGemma3
        model = COCOFTGemma3()
    elif args.model_name == "COCO-FT-Llama-3.2":
        from models import COCOFTLlama3_2
        model = COCOFTLlama3_2()
    elif args.model_name == "COCO-FT-Qwen3-VL":
        from models import COCOFTQwen3VL
        model = COCOFTQwen3VL()
    

    # Setting the prompt
    prompt = "List the objects on the table."

    for scene in metadata:
        scene_id = scene['id']

        # Skip if this scene has already been processed
        if str(scene_id) in overall_scenes_generations:
            print(f"\nImg {scene_id} already processed, skipping.")
            continue

        print(f"\nImg {scene_id}")

        image_path = data_path / scene['image_path']

        scene_generations = scene.copy()

        # Calling the model
        scene_generations['generation'], _, _ = model.run_inference(prompt=prompt, image_path=str(image_path.resolve()))
        print(f"\ngeneration: {scene_generations['generation']}")

        overall_scenes_generations[str(scene_id)] = scene_generations
        # overall_scenes_generations.append(scene_generations)
        
        if scene_id % 10 == 0:
            with open(out_path, "w") as f_out:
                json.dump(overall_scenes_generations,f_out)

    with open(out_path, "w") as f_out:
        json.dump(overall_scenes_generations,f_out)
        

if __name__ == "__main__":
    main()