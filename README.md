# StitchDiffusion

## Setting up
Run in a cmd/any bash terminal
``` python -m venv venv ```
``` venv\Scripts\activate ```
``` pip install -r requirements.txt ```

## To create a pattern
You have to run image_generation.py. You can find all of the possible command line arguments in the code, but just as an example of what was used to generate the cat pattern:
``` python image_generation.py --prompt "cat pixel art, flat colors, minimal shading, simple composition, 40x40 cross stitch design" --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5"--pattern_size 40 --n_colors 10 --num_images 1 --steps 20 --guidance_scale 7.5 --out_dir outputs/cat_pattern ```

That will create a subfolder in your outputs folder with all images. You can also use txt files as prompts if you would like to make more than one image. If that is the case you can run a command like this:

``` python image_generation.py --prompt_file prompts.txt --num_images 2 --pattern_size 48 ```

## Editing the pattern
You'll have to copy the relative path to the pattern image and add is a CL argument, in this case:

``` python crossstitch_editor.py outputs\cat_pattern\sd_1762526686_seed1897366202_p0_pattern_40x40.png```

A pygame window with the pattern will appear where you can edit it yourself and change the colours and the resolution of the image.  

<img width="500" height="400" alt="resolution68x68" src="https://github.com/user-attachments/assets/92921e69-8fa4-4119-9c74-6c4818183999" />  
