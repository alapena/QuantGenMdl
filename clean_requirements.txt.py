# Open the messy file and create a new clean one
with open("requirements.txt", "r") as f_in, open("clean_requirements.txt", "w") as f_out:
    for line in f_in:
        # If the line contains '@', it's a local path; we only want the name
        if " @ " in line:
            package_name = line.split(" @ ")[0]
            f_out.write(f"{package_name}\n") # Write just the name
        else:
            f_out.write(line) # Keep lines that already use '=='