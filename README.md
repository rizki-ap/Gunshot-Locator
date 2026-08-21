# Gunshot-Locator
research about implementation of Gushot Locator 

Product Reference : 
1. Metravib PILAR-V. Link :
2. Boomerang

Publication Reference : 
- main : https://pub.dega-akustik.de/DAGA_1999-2008/data/articles/001903.pdf
- other in /reference

Gunshot Recording Dataset :
- https://zenodo.org/record/7004819 ; Publication : https://doi.org/10.1016/j.dib.2023.109091
- https://cadreforensics.com/audio/
- data in /sound

Target Implementation Component Data :
- in /device

Python Algorithm : (/python)
- physics parameter setup: gunshot_physic.py input:physic_config.ini
- generate clean gunshot signal : gunshot_generate_clean.py ; input:config.ini ; output:*_4ch_clean.wav & *.json
- adding noise to signal : gunshot_add_noise.py ; input:gunshot_clean_4ch.wav ; output:gunshot_noisy_4ch.wav & *.json
- detecting signal tdoa : gunshot_detect_tdoa.py ; input:gunshot_noisy_4ch.wav + *.json
- computing shooter location : gunshot_shooter_locator.py
