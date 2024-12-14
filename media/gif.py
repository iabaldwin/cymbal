from PIL import Image
import glob

# Get list of frames
frames = []
imgs = glob.glob("output_frames/frame_*.png")
imgs.sort()

# Load all frames
for filename in imgs:
    frames.append(Image.open(filename))

# Save as GIF
frames[0].save(
    'output.gif',
    save_all=True,
    append_images=frames[1:],
    duration=33.3,  # 30fps = 33.3ms per frame
    loop=0
)
