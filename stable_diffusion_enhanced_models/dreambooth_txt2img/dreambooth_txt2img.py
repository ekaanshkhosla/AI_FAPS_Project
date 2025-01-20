import os
import time
import torch
import shutil

# List of directories to be removed
directories = [
    "AI_PICS",
    "diffusers",
    "huggingface",
]

for directory in directories:
    if os.path.exists(directory):
        shutil.rmtree(directory)

def make_images(intput_path, coil_defect, num_images):
    HUGGINGFACE_TOKEN = ""
    BRANCH = "main"
    
    ## photo of a el mag coil showing this defect
    instance_prompt = f"image of a electromagnetic coil with {coil_defect} defect"
    class_prompt =  "image of a electromagnetic coil"
    training_steps = 1500
    num_classes = 1
    learning_rate = 2e-6
    output_file = f"AI_PICS/models/my_dreambooth_model_{coil_defect}.safetensors"
    fp16 = True

    CLEAR_LOG = False

    MODEL_NAME="CompVis/stable-diffusion-v1-4"

    OUTPUT_DIR = f"/home/woody/iwfa/iwfa054h/Dreambooth/output_coil_{coil_defect}"
    INSTANCE_DIR = intput_path
    CLASS_DIR = f"/home/woody/iwfa/iwfa054h/Dreambooth/class_coil_{coil_defect}"

    if os.path.exists(CLASS_DIR):
        shutil.rmtree(CLASS_DIR)
    os.makedirs(CLASS_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if 'pipe' in locals():
        del pipe
    
    os.system('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader')

    time_start = time.time()
    def clear():
        from IPython.display import clear_output
        return clear_output()

    os.makedirs('huggingface', exist_ok=True)
    with open('huggingface/token', 'w') as token_file:
        token_file.write(HUGGINGFACE_TOKEN)

    os.system('git clone https://github.com/sagiodev/diffusers.git')
    os.chdir('diffusers')
    os.system('git checkout 08b453e3828f80027d881bb460716af95e192bcd -- ./scripts/convert_diffusers_to_original_stable_diffusion.py')
    os.system('pip install .')

    os.chdir('examples/dreambooth')
    os.system('pip install -r requirements.txt')
    os.system('pip install bitsandbytes')
    os.system("pip install matplotlib")
    import bitsandbytes
    
    os.system(f'accelerate launch train_dreambooth.py \
      --pretrained_model_name_or_path={MODEL_NAME} \
      --revision={BRANCH} \
      --instance_prompt="{instance_prompt}" \
      --class_prompt="{class_prompt}" \
      --class_data_dir={CLASS_DIR} \
      --instance_data_dir={INSTANCE_DIR} \
      --output_dir={OUTPUT_DIR} \
      --with_prior_preservation --prior_loss_weight=1.0 \
      --seed=1337 \
      --resolution=512 \
      --train_batch_size=1 \
      --train_text_encoder \
      --use_8bit_adam \
      --gradient_accumulation_steps=1 \
      --learning_rate={learning_rate} \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --num_class_images={num_classes} \
      --sample_batch_size=4 \
      --max_train_steps={training_steps}')
    
    ckpt_path = '/home/woody/iwfa/iwfa054h/Dreambooth/' + output_file
    dirname = os.path.dirname(ckpt_path)
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.basename(ckpt_path)
    fileanmeWithoutExt = os.path.splitext(filename)[0]
    ExtName = os.path.splitext(filename)[1]
    filenamePattern = fileanmeWithoutExt + '%d' + ExtName
    i = 1
    while os.path.isfile(ckpt_path):
        filename = filenamePattern % i
        ckpt_path = dirname + '/' + filename
        i += 1

    half_arg = "--half" if fp16 else ""
    os.system(f'python /home/woody/iwfa/iwfa054h/Dreambooth/diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py \
        --use_safetensors \
        --model_path {OUTPUT_DIR} \
        --checkpoint_path {ckpt_path} {half_arg}')
    print(f"[*] Converted ckpt saved at {ckpt_path}")
    
    for j in range(0, num_images):
        prompt = f"image of a electromagnetic coil with {coil_defect} defect"
        negative_prompt = ""
        num_samples = 1
        guidance_scale = 5
        num_inference_steps = 50
        height = 512
        width = 512 
        seed = j

        os.chdir('/home/woody/iwfa/iwfa054h/Dreambooth/')
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        from matplotlib.pyplot import figure, imshow, axis
        from matplotlib.image import imread
        import numpy as np

        if 'pipe' not in locals():
            pipe = StableDiffusionPipeline.from_pretrained(OUTPUT_DIR, safety_checker=None, torch_dtype=torch.float16).to("cuda")
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            g_cuda = None

        g_cuda = torch.Generator(device='cuda')
        g_cuda.manual_seed(seed)

        save_directory = f'{coil_defect}_{training_steps}_{num_classes}_{learning_rate}_{guidance_scale}_{num_inference_steps}'
        os.makedirs(save_directory, exist_ok=True)

        with torch.autocast("cuda"), torch.inference_mode():
            images = pipe(
                prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=g_cuda
            ).images

            for i, im in enumerate(images):
                im.save(os.path.join(save_directory, f'{coil_defect}_{j}.jpg'))
                

defects_list = [
    ['/home/woody/iwfa/iwfa054h/Dreambooth/data_seperated_10%_filtered_images_resized/first_view/all_defect', 'First_view_all_defect', 421],
    ['/home/woody/iwfa/iwfa054h/Dreambooth/data_seperated_10%_filtered_images_resized/first_view/gap_double', 'First_view_gap_double', 77],
    ['/home/woody/iwfa/iwfa054h/Dreambooth/data_seperated_10%_filtered_images_resized/first_view/no_defect', 'First_view_no_defect', 1205],
    ['/home/woody/iwfa/iwfa054h/Dreambooth/data_seperated_10%_filtered_images_resized/first_view/only_gap', 'First_view_only_gap', 151],

    ['/home/woody/iwfa/iwfa054h/Dreambooth/data_seperated_10%_filtered_images_resized/second_view/all_defect', 'second_view_all_defect', 250],
    ['/home/woody/iwfa/iwfa054h/Dreambooth/data_seperated_10%_filtered_images_resized/second_view/gap_double', 'second_view_gap_double', 155],
    ['/home/woody/iwfa/iwfa054h/Dreambooth/data_seperated_10%_filtered_images_resized/second_view/no_defect', 'second_view_no_defect', 1091],
    ['/home/woody/iwfa/iwfa054h/Dreambooth/data_seperated_10%_filtered_images_resized/second_view/only_double', 'second_view_only_double', 2],
    ['/home/woody/iwfa/iwfa054h/Dreambooth/data_seperated_10%_filtered_images_resized/second_view/only_gap', 'second_view_only_gap', 434]
]

for defect in defects_list:
    make_images(*defect)
