import os
from pathlib import Path
import shutil
import zipfile
import urllib.parse
import urllib.request


if __name__ == "__main__":

    path = Path("temp")
    path.mkdir(exist_ok=True)
    for file in [
        "abaqus-user-subroutine.cpp",
        "ExtraUtilities.h",
        "abaqus_v6.env",
    ]:
        shutil.copyfile(file, path / file)
    os.chdir(path)

    # Install libtorch
    if not os.path.exists("./libtorch"):
        file = "libtorch-win-shared-with-deps-2.4.0+cpu.zip"
        url = f"https://download.pytorch.org/libtorch/cpu/" + urllib.parse.quote(file, safe='')
        urllib.request.urlretrieve(url, file)
        zipfile.ZipFile(file).extractall()
    os.environ["PATH"] = ";".join([os.environ["PATH"], os.path.abspath("./libtorch/lib")])

    # Prepare the *.inp file for Abaqus
    job = "uelnonli"
    inp = job + ".inp"
    url = f"https://help.3ds.com/2020/English/DSSIMULIA_Established/SIMAINPRefResources/{inp}"
    urllib.request.urlretrieve(url, inp)

    # Run the job
    cpp = "abaqus-user-subroutine.cpp"
    ifort_env = r'"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat" intel64'
    cmd = ifort_env + "&" + f"abaqus int job={job} user={cpp}"
    os.system(cmd)