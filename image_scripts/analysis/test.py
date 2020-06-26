from PIL import Image,ImageDraw,ImageFont;
import datetime;

timestamp=datetime.datetime.now();
fimage='last_update.png';
image=Image.new("RGBA",(3000,240),(255,255,255));
draw=ImageDraw.Draw(image);
font=ImageFont.truetype("../etc/fonts/vera_sans/Vera.ttf",244);

draw.text((0,0), timestamp.strftime("%a, %b %d at %I:%M%p"), (0,0,0), font=font);
image_resized=image.resize((3000,240),Image.ANTIALIAS);
image_resized.save(fimage);
