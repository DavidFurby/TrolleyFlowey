# TrolleyFlowey
AI solution to improve trolley
Trolley Flowey

Improve the schedule of trolley transportation by predicting demand using variables like time of day, traffic, city events, etc to increase or decrease the number of ongoing trolleys on different routes dynamically.


 This project aims to optimize the allocation of available resources in trolley transportation by predicting sudden changes in traffic and other areas. It is not intended to solve the issue of limited resources but to suggest possible alternatives for the spare resources that are available.  My personal motivation for this idea is that my own route was recently halved in its ongoing trolleys, resulting in almost guaranteed overcrowding during specific hours and less convenient scheduling.


## How it works
The tool will identify routes that might be over or underrepresented based on data analysis and may be used to suggest changes in the future use of these routes. Its primary function is to quickly identify routes that might demand more resources in real time. It can take into consideration the weather, what type of day it is and what popular destinations exist on the routes. Knowing the weather by itself won't tell us much. If it's bad weather then we can assume that all routes are going to be more popular since people will decide to go by trolley instead of walking. But knowing possible destinations combined with the weather and maybe the day might tell us where people are more likely to go. If it's sunny and the weekend, it's more likely the route going to an amusement park will be more popular. If it's raining and almost Christmas, it's likely that more people will go to the shopping mall. 
Resources: By ‘resources’, it refers not only to trolleys but also to temporary bus routes or extra personnel, depending on the demand of a particular route at a given time.
Assumptions: If one route is to have increased resources, another should decrease correspondingly. Weather might not be a significant variable, as it’s assumed that weather conditions will be the same across the entire transportation area. Therefore, if bad weather were a factor in increasing resources, those added resources would presumably be needed for all other routes as well.





```
Def main():

Routes = [{routeId = 1, activeTrolleys = 2, demand= normal}];


RouteActivityPerHour = [{hour = "08:00", averagePassengerAmountPerTrip=300}]


Data sources:

Google Maps Platform Documentation  |  Distance Matrix API  |  Google for Developers
Weather API - OpenWeatherMap

Initially, the project will be using general APIs:s to make predictions. Avoiding API:S for specific trolley services (like the one for my city) means it's going to have a more dynamic use that can be applied in other circumstances. 

Where does your data come from? Do you collect it yourself or do you use data collected by someone else?



## Challenges
This tool does not aim to increase the total resources available but rather to make better use of existing resources by redistributing them based on demand. It's also to be expected that the project should be used as suggestions and not requirements. Every type of variable can't be considered, and therefore the final decision should always be made by the person/people in charge. It will also have to be rather dynamic to handle all the possible reasons why a route might be more popular at certain times.

## What next?
As Trolley Flowey gains experience and handles more data, it can be utilized to improve the scheduling of trolleys. Its primary function should remain the identification of sudden shifts in day-to-day traffic that might necessitate resource reallocation. However, using this data to determine future schedules would require more generalized data, which might not be within Trolley Flowey’s capabilities. Therefore, while TrolleyFlowey can contribute to short-term resource management, its application in long-term planning may be limited.

## Acknowledgments
Bing AI
