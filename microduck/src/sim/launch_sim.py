"""Launch the local viewer with the Python shared-library path on macOS."""
import os
from pathlib import Path
import sys
import sysconfig


def launch_environment():
    env = dict(os.environ)
    if sys.platform == "darwin":
        libdir = sysconfig.get_config_var("LIBDIR")
        library = sysconfig.get_config_var("LDLIBRARY")
        if libdir and library and (Path(libdir) / library).is_file():
            previous = env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
            paths = list(dict.fromkeys([libdir, *filter(None, previous.split(":"))]))
            env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(paths)
    return env


def main():
    executable = (Path(sys.executable).parent / "mjpython"
                  if sys.platform == "darwin" else Path(sys.executable))
    script = Path(__file__).with_name("sim_main.py")
    os.execvpe(str(executable), [str(executable), str(script), *sys.argv[1:]], launch_environment())


if __name__ == "__main__":
    main()
