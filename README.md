# EMRI-s
Code concerning EMRI signal noise studies

Un_Geometrized.ipbyn current progress on calculating the SNR values for evolving EMRI system based on Peters-Matthew equations of orbital evolution. Since they are written in normal units, I left them as such and evolved to calculate the SNR and characteristic strain graphs.

Barack_Cutler_Geometrized.ipbyn constains code evolving orbital parameters defined in equations 17 - 31 of Barack adn Culter (2003) paper. Contains geometrized units of seconds since Mass*frequency must be unitless, thus Mass = G*M/c^3, meaning Mass = meters^3*kg*s^3/kg*s^2*meters^3 = seconds

-> Consistent geometric units have been implemented
-> Evolving the two different versions of the PN equations separately as two different definitions of the same inspiral. 

Issues: The scale for the SNR and characteristic strain graphs are still slightly off, but only about one order of magnitude. They do have the correct behaviour expected to recreate figures 2 & 3 in Pau's paper. Issues could come from the difference in calcualting just Peters-Matthew equations instead of Barack/Cutler equations, but I believe there is also something else going on since the values for eccentricity at a given time (which is the main use of the PN equations) look very close based on Figures 3 from Pau's graph. 
