# Severity personal injury accidents prediction challenge

**Credits to : Camille Cochet - Matthieu Colin de Verdière - Mithuran Gajendran - Arnaud Houzé de l'Aulnoit - Hugo Mallet**

Each year thousands of traffic accidents happen in France what can cause injuries or even the death to the involved users. Modeling accident severity prediction critical to the effective performance of road traffic systems for improved safety. Moreover this feature is important for countries to demonstrate the quality and the security of their infrastructures.

Security is also a very important subject for car manufacturers which are asked to respect many constraints and do many tests to be allowed to sell a new car.

Otherwise assurance companies are also interested in this area because knowing the risk of accident gravity help them to adjust their assurance contract prices regarding the type of car and the experience of the driver.

## Purpose

In this data challenge you are expected to determine the severity of traffic accident involving injuries in 2018 in the French territory. The aim is to predict for each involved person in the accident the level of injuries between four possibilities :
<ol>
    <li> Uninjured
    <li> Killed
    <li> Inpatient injury
    <li> Minor injury
</ol>

To do that you dispose of 4 csv files that you have to pre-process using Python librairies like Scikit-Learn:
* `caracteristiques-2017.csv`  -  Describes the general circumstances of the accident
* `lieux-2017.csv`  -  Describes the main location of the accident
* `vehicules-2017.csv`  -  Indicates the involved vehicles
* `usagers-2017.csv`  -  Indicates the involved users

You are also expected to use Machine Learning tools of your choice to make your predictions.
