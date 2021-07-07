rm data/test_samemaps_keyboard.arff
rm data/training_keyboard.arff
rm data/test_othermaps_tutorial1.arff
rm data/test_samemaps_tutorial1.arff
rm data/training_git .arff

# Training keyboard-------------------------------------------------------------------------------
python3 busters.py -g RandomGhost -l layouts/oneHunt.lay -r data/training_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/bigHunt.lay -r data/training_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/20Hunt.lay -r data/training_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/openHunt.lay -r data/training_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/classic.lay -r data/training_keyboard.arff
python3 busters.py -g RandomGhost -l layouts/oneHunt.lay -r data/training_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/bigHunt.lay -r data/training_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/20Hunt.lay -r data/training_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/openHunt.lay -r data/training_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/classic.lay -r data/training_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/1.lay -r data/training_keyboard.arff -k 1 
python3 busters.py -g RandomGhost -l layouts/2.lay -r data/training_keyboard.arff -k 1 
python3 busters.py -g RandomGhost -l layouts/3.lay -r data/training_keyboard.arff -k 1 
python3 busters.py -g RandomGhost -l layouts/4.lay -r data/training_keyboard.arff -k 1 
python3 busters.py -g RandomGhost -l layouts/1.lay -r data/training_keyboard.arff -k 1 
python3 busters.py -g RandomGhost -l layouts/2.lay -r data/training_keyboard.arff -k 1 
python3 busters.py -g RandomGhost -l layouts/3.lay -r data/training_keyboard.arff -k 1 
python3 busters.py -g RandomGhost -l layouts/4.lay -r data/training_keyboard.arff -k 1 


#testing samemaps
python3 busters.py -g RandomGhost -l layouts/oneHunt.lay -r data/test_samemaps_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/bigHunt.lay -r data/test_samemaps_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/20Hunt.lay -r data/test_samemaps_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/openHunt.lay -r data/test_samemaps_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/classic.lay -r data/test_samemaps_keyboard.arff
python3 busters.py -g RandomGhost -l layouts/1.lay -r data/test_samemaps_keyboard.arff -k 1
python3 busters.py -g RandomGhost -l layouts/2.lay -r data/test_samemaps_keyboard.arff -k 1
python3 busters.py -g RandomGhost -l layouts/3.lay -r data/test_samemaps_keyboard.arff -k 1
python3 busters.py -g RandomGhost -l layouts/4.lay -r data/test_samemaps_keyboard.arff -k 1

#testing othermaps
python3 busters.py -g RandomGhost -l layouts/sixHunt.lay -r data/test_othermaps_keyboard.arff
python3 busters.py -g RandomGhost -l layouts/smallHunt.lay -r data/test_othermaps_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/newmap.lay -r data/test_othermaps_keyboard.arff
python3 busters.py -g RandomGhost -l layouts/customLayout.lay -r data/test_othermaps_keyboard.arff 
python3 busters.py -g RandomGhost -l layouts/testClassic.lay -r data/test_othermaps_keyboard.arff

# Training agent -------------------------------------------------------------------------------
python3 busters.py -g RandomGhost -l layouts/oneHunt.lay -p BasicAgentAA -r data/training_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/bigHunt.lay -p BasicAgentAA -r data/training_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/20Hunt.lay -p BasicAgentAA -r data/training_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/openHunt.lay -p BasicAgentAA -r data/training_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/classic.lay -p BasicAgentAA -r data/training_tutorial1.arff
python3 busters.py -g RandomGhost -l layouts/oneHunt.lay -p BasicAgentAA -r data/training_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/bigHunt.lay -p BasicAgentAA -r data/training_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/20Hunt.lay -p BasicAgentAA -r data/training_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/openHunt.lay -p BasicAgentAA -r data/training_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/classic.lay -p BasicAgentAA -r data/training_tutorial1.arff
python3 busters.py -g RandomGhost -l layouts/1.lay -p BasicAgentAA -r data/training_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/2.lay -p BasicAgentAA -r data/training_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/3.lay -p BasicAgentAA -r data/training_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/4.lay -p BasicAgentAA -r data/training_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/1.lay -p BasicAgentAA -r data/training_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/2.lay -p BasicAgentAA -r data/training_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/3.lay -p BasicAgentAA -r data/training_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/4.lay -p BasicAgentAA -r data/training_tutorial1.arff -k 1


#testing samemaps
python3 busters.py -g RandomGhost -l layouts/oneHunt.lay -p BasicAgentAA -r data/test_samemaps_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/bigHunt.lay -p BasicAgentAA -r data/test_samemaps_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/20Hunt.lay -p BasicAgentAA -r data/test_samemaps_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/openHunt.lay -p BasicAgentAA -r data/test_samemaps_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/classic.lay -p BasicAgentAA -r data/test_samemaps_tutorial1.arff
python3 busters.py -g RandomGhost -l layouts/1.lay -p BasicAgentAA -r data/test_samemaps_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/2.lay -p BasicAgentAA -r data/test_samemaps_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/3.lay -p BasicAgentAA -r data/test_samemaps_tutorial1.arff -k 1
python3 busters.py -g RandomGhost -l layouts/4.lay -p BasicAgentAA -r data/test_samemaps_tutorial1.arff -k 1

#testing othermaps
python3 busters.py -g RandomGhost -l layouts/sixHunt.lay -p BasicAgentAA -r data/test_othermaps_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/smallHunt.lay -p BasicAgentAA -r data/test_othermaps_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/newmap.lay -p BasicAgentAA -r data/test_othermaps_tutorial1.arff
python3 busters.py -g RandomGhost -l layouts/customLayout.lay -p BasicAgentAA -r data/test_othermaps_tutorial1.arff 
python3 busters.py -g RandomGhost -l layouts/testClassic.lay -p BasicAgentAA -r data/test_othermaps_tutorial1.arff
