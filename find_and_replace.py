import os

for root, dirs, files in os.walk("."):
    for file in files:
        file_path = os.path.join(root, file)
        print(str(file_path))
        try:
            with open(file_path, "r") as f:
                try:
                    lines = f.read()
                except:
                    print(str(file_path) + " failed")
            lines = lines.replace("dosenet/radwatch-airmonitor", "dosenet/radwatch-airmonitor/radwatch-airmonitor")
            with open(file_path, "w") as f:
                f.write(lines)
        except:
            print(str(file_path) + " failed")
