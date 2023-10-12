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
import numpy as np

def main():
    Demand = {'low': 0.9, 'normal': 1, 'high': 1.1}
    Weather = {'Sunny': 1.1, 'Cloudy': 1, 'Rainy':0.9}
    Routes = [{'routeId': 1, 'demand':'normal', 'destinations':[{'name': "amusement park", 'outdoors': True, 'demand': "high"}, {'name': "Museum", 'outdoors': False, 'demand': "low"}]}]
    weather_today =  {'mostLikely': "Sunny"}

    # Training data
    training_data = [
        {'weather': 'Sunny', 'routeId': 1, 'destinations': [{'name': "amusement park", 'outdoors': True, 'demand': "high"}, {'name': "Museum", 'outdoors': False, 'demand': "low"}], 'demand':'normal', 'actual_demand': 1.2},
        {'weather': 'Cloudy', 'routeId': 2, 'destinations': [{'name': "Zoo", 'outdoors': True, 'demand': "normal"}], 'demand':'high', 'actual_demand': 0.9},
        {'weather': 'Rainy', 'routeId': 3, 'destinations': [{'name': "Mall", 'outdoors': False, 'demand': "high"}, {'name': "Aquarium", 'outdoors': False, 'demand': "normal"}], 'demand':'low', 'actual_demand': 0.8},
        # Add more training examples here
    ]

    test_data = [{'weather' : weather_today['mostLikely'], 
                  'routeId' : route['routeId'], 
                  'demand' : route['demand'], 
                  'destinations' : route['destinations']} for route in Routes]    

    def hidden_activation(z):
        # ReLU activation.
        return np.maximum(0, z)

    def output_activation(z):
        # identity (linear) activation.
        return z
    
    # Define weights
    routeWeighs = np.random.rand(len(test_data))
    weatherWeighs = np.random.rand();
    weightChange = 0.2
    
    for turns in range(10):
        for i, route in enumerate(test_data):
            demands = []
            for j, destination in enumerate(route['destinations']):
                weather_multiplier = int(destination['outdoors']) * Weather[weather_today['mostLikely']]
                demand = Demand[destination['demand']] * (weather_multiplier * weatherWeighs)
                demands.append(demand)
            averages = [np.mean(demand) for demand in demands]
            h1_in = np.dot(Demand[route['demand']] * routeWeighs[i], averages)
            h1_out = hidden_activation(h1_in)
            out = output_activation(h1_out)
            error = out - training_data[i]['actual_demand']
            print(error[i])
            if error[i] < 0:
                routeWeighs += weightChange
                weatherWeighs += weightChange
            if error[i] > 0:
                routeWeighs -= weightChange
                weatherWeighs -= weightChange
        
        

main()
```




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
