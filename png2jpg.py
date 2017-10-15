from PIL import Image
import glob, os
os.chdir("TestVal")
for file in glob.glob("*.png"):
	im = Image.open("Abs/path/to/folder" + file)
	im.convert('RGB').save(file + ".jpg","JPEG")