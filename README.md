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
- gen_config.ini : all parameter needed to configure generation of real gunshot signal
- gs_gen_physic.py : physics parameter setup, 
- gs_gen_clean_signal.py : generate clean gunshot signal in multichannel wav file and *.json 
- gs_gen_add_noise.py : adding noise to signal, create multichannel wav file and *.json
- gs_gen_apply_adc.py : adding noise in mics, afe and adc.
- det_config.ini : all parameter needed to configure detection gunshot shooter origin
- gs_det_signal_prepare.py :detecting sw & mb peak, normalization
- gs_det_tdoa.py :doing gcc-phat algoritm, create 6 pair tDoA
- gs_det_shooter_locator.py : computing hyperboloid equation, determine shooter location
- gs_det_classify_bullet.py : determine bullet property, mach, caliber, etc
- util_get_accoustic_param.py : extracting accoustic parameter, include noise from wav file
- util_plot_signal.py : plotting signal in multichannel wav file

FPGA rtl implementation :
- in /verilog

