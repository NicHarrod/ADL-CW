# ADL-CW
Applied Deep Learning Coursework, looking at recipes with EPIC-HD

Main runcode: SIAMESETEST.py

Scripts:
python SIAMESETEST.py --resume_dir best_models/Improved_Model.pth --epochs 0
This will generate our best results for the test set

python SIAMESETEST.py --resume_dir best_models/Base_Model.pth --epochs 0
This will generate our best results just using the base model and hyperparameters

Within our code we have indicated what is from the base model and what improvements we include, as well as some commented out improvements that didn't pan out


For a standard run, our default hyperparameters are the best ones we decided on so simply python SIAMESETEST.py should work

For any questions please email us!
os22128 (Nic)
xd22898 (Valentina)