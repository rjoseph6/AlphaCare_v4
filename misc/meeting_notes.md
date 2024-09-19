# Meeting Notes

These are all the notes taken during the meetings with the advisor.

# 09-05-24

First meeting of the semester. Our next meeting is scheduled for next Thursday (09-12-24) same time, 10 AM. Goal for next meeting is to have a working barebone prototype. 
- User can upload image
- simple ml model inference
- model prediction displayed
- heatmap of image

The Web App should have clear installation instructions, enough so a lay person could run the app. We debated the utility of the HeatMap. Overall we came to the decision a more Hierarchial approach would do a better job than the heatmap can accomplish. 

### SECOND DEADLINE (2 WEEKS)

The second deadline we discussed having is a finalized fully trained model. We had a discussion about class weights. Advisor wants me to have complete understanding about every tool and framework I use. So moving forward I am going to attempt to use less things that I already fully understand. Essentially saying use less ChatGPT. 

### HIERARCHIAL VS ONE 

We also discussed Hierarchial Neural Networks. I made the case that a simple massive NN could do a better job than an hierarchial NN because it will be able to capture more complex relationships directly, without the need for redefined hierarchies. I compared it to the Tesla Full Self Driving stack. Tesla turned theer 100 ML models into simply just one massive model that takes photons in (images) and outputs only steering and pedal controls. Check out this super unknown talk by a former Tesla AI engineer where he explains the idea. 
https://youtu.be/OKDRsVXv49A?si=bDnlsWGepy4a2QO4&t=918 

# 0 9 - 1 2 - 2 4
After today's meeting, I am going to focus on the following: explainable AI and hierarchial AI. By the next meeting we should have the SOTA (State of the Art) model trained and ready to go. I am likely going to combine multiple datasets to get the best possible model. Advisor said I should learn the diagnostic procedures for skin cancer so I can better understand the data I am working with. I should understand the difference between the different types of skin cancer and how doctors normally diagnose them. For the UI, he wants me to give the user more information then just the diagnosis. For example, if the model predicts Melanoma it would have a link to a page that explains what Melanoma is and what the signs for it are (knowledge base). NEVER USE HEAT MAP AGAIN UNTIL FULLY UNDERSTAND IT AND CODE ONE FROM SCRATCH. The most important addition he wants me to add is a hierarchial model. He wants me to have a model that can take in an image and output a diagnosis. Then take that diagnosis and output a more specific diagnosis. For example, if the first model outputs Melanoma, the second model should output the specific type of Melanoma.

![](../ui/hierarchy.png)

# 09-19-24

This weeks meeting objectives:
- Have a working Hierarchial model (3-tier)
- Review skin disease diagnosis (skin_diseases.md)
- Have a fully trained state of the art model (ResNet18)

Next Meeting Objectives:
- Finalize the UI using design software (continuity between disease types)
- Modify ui to new design (ui_v3)
- Finalize first disease model and its ui