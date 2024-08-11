import os
from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as req_file:
        return [line.strip() for line in req_file if line.strip() and not line.startswith('#')]

setup(
    name="pyannote-whisper",
    py_modules=["pyannote-whisper"],
    version="1.0",
    description="Speech Recognition plus diarization",
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    author="Ruiqing",
    url="https://github.com/yinruiqing/pyannote-whisper",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'pyannote-whisper=pyannote_whisper.cli.transcribe:cli',
            'pyannote-whisper-serve=pyannote_whisper.serve:main'
        ],
    },
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
