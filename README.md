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
    np.random.seed(0)  # Set a random seed for reproducibility

    Demand = {'low': 0.9, 'normal': 1, 'high': 1.1}
    Weather = {'Sunny': 1.1, 'Cloudy': 1, 'Rainy':0.9}
    Routes = [
        {'routeId': 1, 'expected_demand':'normal', 'destinations':[{'name': "amusement park", 'outdoors': True, 'expected_demand': "high"}, {'name': "Museum", 'outdoors': False, 'expected_demand': "low"}]},
        {'routeId': 2, 'expected_demand':'high', 'destinations':[{'name': "Zoo", 'outdoors': True, 'expected_demand': "normal"}]},
        {'routeId': 3, 'expected_demand':'low', 'destinations':[{'name': "Mall", 'outdoors': False, 'expected_demand': "high"}, {'name': "Aquarium", 'outdoors': False, 'expected_demand': "normal"}]}
    ]
    weather_today =  {'mostLikely': "Sunny"}

    # Training data
    training_data = [
        {'weather': 'Sunny', 'routeId': 1, 'destinations':[{'name': "amusement park", 'outdoors': True, 'expected_demand': "high"}, {'name': "Museum", 'outdoors': False, 'expected_demand': "low"}], 'expected_demand':'normal', 'actual_demand': 1.2},
        {'weather': 'Cloudy', 'routeId': 2, 'destinations':[{'name': "Zoo", 'outdoors': True, 'expected_demand': "normal"}], 'expected_demand':'high', 'actual_demand': 0.9},
        {'weather': 'Rainy', 'routeId': 3, 'destinations':[{'name': "Mall", 'outdoors': False, 'expected_demand': "high"}, {'name': "Aquarium", 'outdoors': False, 'expected_demand': "normal"}], 'expected_demand':'low', 'actual_demand': 0.8},
        # Add more training examples here
    ]


    test_data = [{'weather' : weather_today['mostLikely'], 
              'routeId' : route['routeId'], 
              'expected_demand' : route['expected_demand'], 
              'destinations' : route['destinations']} for i,route in enumerate(Routes)]
                  
    def hidden_activation(z):
        # ReLU activation.
        return np.maximum(0, z)

    def output_activation(z):
        # identity (linear) activation.
        return z

    def mse_loss(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def mse_loss_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true)
    
    # Define weights
    routeWeights = np.random.rand(len(test_data))
    weatherWeighs = np.random.rand()
    learning_rate = 0.1
    
    for turns in range(1000):
        grads = []  # Reset gradients at the start of each epoch
        for i, route in enumerate(training_data):
            listOfDemands = []
            for j, destination in enumerate(route['destinations']):
                weather_multiplier = int(destination['outdoors']) * Weather[weather_today['mostLikely']]
                demand = Demand[destination['expected_demand']] * (weather_multiplier * weatherWeighs)
                listOfDemands.append(demand)
            error = 0
            h1_in = Demand[route['expected_demand']] * routeWeights[i] * np.mean(listOfDemands) 
            h1_out = hidden_activation(h1_in)
            out = output_activation(h1_out)
            error = mse_loss(training_data[i]['actual_demand'], out)
            derror_dout = mse_loss_derivative(training_data[i]['actual_demand'], out)
            dout_dh1out = 1
            dh1out_dh1in = h1_out > 0
            dh1in_dw = demand
            grad = derror_dout * dout_dh1out * dh1out_dh1in * dh1in_dw
            grads.append(grad)
            print("ID:", route['routeId'])
            print('Actual demand:', training_data[i]['actual_demand'])
            print('Predicted demand:', out)
            print('Error:', error)
            # Update weights using average gradient
            routeWeights -= learning_rate * np.mean(grads)
            weatherWeighs -= learning_rate * np.mean(grads)
        # After training the model
    for i, route in enumerate(test_data):
        listOfDemands = []
        for j, destination in enumerate(route['destinations']):
            weather_multiplier = int(destination['outdoors']) * Weather[weather_today['mostLikely']]
            demand = Demand[destination['expected_demand']] * (weather_multiplier * weatherWeighs)
            listOfDemands.append(demand)
        h1_in = Demand[route['expected_demand']] * routeWeights[i] * np.mean(listOfDemands) 
        h1_out = hidden_activation(h1_in)
        out = output_activation(h1_out)
        print("ID:", route['routeId'])
        print('Predicted demand:', out)



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
