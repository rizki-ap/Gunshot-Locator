Python Algorithm :
- physics parameter setup: gunshot_physic.py input:physic_config.ini
- generate clean gunshot signal : gunshot_generate_clean.py ; input:config.ini ; output:*_4ch_clean.wav & *.json
- adding noise to signal : gunshot_add_noise.py ; input:gunshot_clean_4ch.wav ; output:gunshot_noisy_4ch.wav & *.json
- detecting signal tdoa : gunshot_detect_tdoa.py ; input:gunshot_noisy_4ch.wav + *.json
- computing shooter location : gunshot_shooter_locator.py

Other old ipynb : /old

utility : /util
- plot signal in multichannel wav file : 
