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

    class Destination:
        def __init__(self, name, outdoors, attraction_type, expected_demand):
            self.name = name
            self.outdoors = outdoors
            self.attraction_type = attraction_type
            self.expected_demand = expected_demand

    class Route:
        def __init__(self, route_id, expected_demand, destinations):
            self.route_id = route_id
            self.expected_demand = expected_demand
            self.destinations = [destination for destination in destinations]

    Demand = {"low": 0.9, "normal": 1, "high": 1.1}
    Weather = {"Sunny": 1.1, "Cloudy": 1, "Rainy": 0.9}
    AttractionType = {"Leisure": 1.1, "Both": 1, "Work": 0.9}
    Routes = [
        Route(
            1,
            Demand["high"],
            [
                Destination(
                    "amusement park", True, AttractionType["Leisure"], Demand["high"]
                ),
                Destination("Museum", False, AttractionType["Leisure"], Demand["low"]),
            ],
        ),
        Route(
            2,
            Demand["high"],
            [
                Destination("Zoo", True, AttractionType["Leisure"], Demand["normal"]),
            ],
        ),
        Route(
            3,
            Demand["low"],
            [
                Destination("Mall", False, AttractionType["Both"], Demand["high"]),
                Destination(
                    "Aquarium", False, AttractionType["Leisure"], Demand["normal"]
                ),
            ],
        ),
    ]

    weather_today = {"mostLikely": "Sunny"}
    dayOfWeek_today = {"dayOfWeek": "Weekend"}

    # Training data
    training_data = [
        {
            "weather": "Sunny",
            "dayOfWeek": "Weekend",
            "route": Route(
                1,
                "normal",
                [
                    Destination(
                        "Amusement park",
                        True,
                        AttractionType["Leisure"],
                        Demand["high"],
                    ),
                    Destination("Museum", False, AttractionType["Both"], Demand["low"]),
                ],
            ),
            "actual_demand": 1.2,
        },
        {
            "weather": "Cloudy",
            "dayOfWeek": "Weekday",
            "route": Route(
                2,
                "high",
                [
                    Destination(
                        "Zoo", True, AttractionType["Leisure"], Demand["normal"]
                    ),
                ],
            ),
            "actual_demand": 0.9,
        },
        {
            "weather": "Rainy",
            "dayOfWeek": "Weekday",
            "route": Route(
                3,
                "low",
                [
                    Destination("Mall", False, AttractionType["Both"], Demand["high"]),
                    Destination(
                        "Aquarium", False, AttractionType["Leisure"], Demand["normal"]
                    ),
                ],
            ),
            "actual_demand": 0.8,
        },
        {
            "weather": "Sunny",
            "dayOfWeek": "WeekEnd",
            "route": Route(
                4,
                "high",
                [
                    Destination(
                        "Beach", True, AttractionType["Leisure"], Demand["high"]
                    ),
                    Destination(
                        "Cafe", False, AttractionType["Both"], Demand["normal"]
                    ),
                ],
            ),
            "actual_demand": 1.3,
        },
        {
            "weather": "Cloudy",
            "dayOfWeek": "WeekEnd",
            "route": Route(
                5,
                "normal",
                [Destination("Park", True, AttractionType["Leisure"], Demand["low"])],
            ),
            "actual_demand": 0.95,
        },
        {
            "weather": "Rainy",
            "dayOfWeek": "Weekday",
            "route": Route(
                6,
                "high",
                [
                    Destination(
                        "Cinema", False, AttractionType["Leisure"], Demand["high"]
                    ),
                    Destination(
                        "Restaurant", False, AttractionType["Leisure"], Demand["high"]
                    ),
                    Destination(
                        "University", False, AttractionType["Work"], Demand["high"]
                    ),
                ],
            ),
            "actual_demand": 1.1,
        },
    ]

    test_data = [
        {
            "weather": weather_today["mostLikely"],
            "dayOfWeek": dayOfWeek_today["dayOfWeek"],
            "route": route,
        }
        for route in Routes
    ]

    def hidden_activation(z):
        # ReLU activation.
        return np.maximum(0, z)

    def output_activation(z):
        # identity (linear) activation.
        return z

    def mse_loss_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true)

    # Define weights
    hidden1_weights = np.random.rand(2, 2)
    hidden2_weights = np.random.rand(2, 2)
    hidden3_weights = np.random.rand(2, 1)
    learning_rate = 0.001

    def update_weights(
        learning_rate,
        hidden1_grads,
        hidden2_grads,
        hidden3_grads,
        hidden1_weights,
        hidden2_weights,
        hidden3_weights,
    ):
        hidden1_weights -= learning_rate * np.mean(hidden1_grads)
        hidden2_weights -= learning_rate * np.mean(hidden2_grads)
        hidden3_weights -= learning_rate * np.mean(hidden3_grads)
        return hidden1_weights, hidden2_weights, hidden3_weights

    def calculate_gradient(
        actual_demand,
        out,
        h2_out,
        h1_out,
        h1_weight,
        h2_weight,
        h3_weight,
    ):
        # Calculate error
        error_out = mse_loss_derivative(actual_demand, out)
        # Calculate derivatives
        dh1out_dh1in = (h1_out > 0).astype(float)  # derivative of ReLU
        dh2out_dh2in = (h2_out > 0).astype(float)  # derivative of ReLU

        # Calculate gradients
        hidden0_grad = (
            error_out * h1_weight * h3_weight * dh2out_dh2in * h2_weight * dh1out_dh1in
        )
        hidden2_grad = error_out * h3_weight * dh2out_dh2in * h2_weight
        hidden3_grad = error_out * h3_weight
        return hidden0_grad, hidden2_grad, hidden3_grad

    def predict_demand(data, hidden1_weights, hidden2_weights, hidden3_weights):
        weather_input = (
            np.mean(
                [
                    int(destination.outdoors) * Weather[data["weather"]]
                    for destination in data["route"].destinations
                ]
            )
            if data["route"].destinations
            else 0
        )
        destination_demand_input = (
            np.mean(
                [
                    destination.expected_demand
                    for destination in data["route"].destinations
                ]
            )
            if data["route"].destinations
            else 0
        )
        # Create an input array with the averages
        input_array = np.array([weather_input, destination_demand_input])

        h1_in = np.dot(input_array, hidden1_weights)
        h1_out = hidden_activation(h1_in)
        h2_in = np.dot(h1_out, hidden2_weights)
        h2_out = hidden_activation(h2_in)
        out_in = np.dot(h2_out, hidden3_weights)
        out = output_activation(out_in)
        return out, h1_out, h2_out

    def train_model(
        training_data,
        learning_rate,
        hidden1_weights,
        hidden2_weights,
        hidden3_weights,
    ):
        for turns in range(100):
            hidden1_grad_list = []
            hidden2_grad_list = []
            hidden3_grad_list = []

            for data in training_data:
                out, h1_out, h2_out = predict_demand(
                    data,
                    hidden1_weights,
                    hidden2_weights,
                    hidden3_weights,
                )

                (
                    hidden1_grad,
                    hidden2_grad,
                    hidden3_grad,
                ) = calculate_gradient(
                    data["actual_demand"],
                    out,
                    h2_out,
                    h1_out,
                    hidden1_weights,
                    hidden2_weights,
                    hidden3_weights,
                )

                hidden1_grad_list.append(hidden1_grad)
                hidden2_grad_list.append(hidden2_grad)
                hidden3_grad_list.append(hidden3_grad)

                (
                    hidden1_weights,
                    hidden2_weights,
                    hidden3_weights,
                ) = update_weights(
                    learning_rate,
                    hidden1_grad_list,
                    hidden2_grad_list,
                    hidden3_grad_list,
                    hidden1_weights,
                    hidden2_weights,
                    hidden3_weights,
                )

        return hidden1_weights, hidden2_weights, hidden3_weights

    def test_model(test_data, hidden1_weights, hidden2_weights, hidden3_weights):
        for data in test_data:
            (
                out,
                _,
                _,
            ) = predict_demand(data, hidden1_weights, hidden2_weights, hidden3_weights)
            print("ID:", data["route"].route_id)
            print("Predicted demand:", out)

    # Train the model
    hidden1_weights, hidden2_weights, hidden3_weights = train_model(
        training_data,
        learning_rate,
        hidden1_weights,
        hidden2_weights,
        hidden3_weights,
    )
    # Test the model
    test_model(
        test_data,
        hidden1_weights,
        hidden2_weights,
        hidden3_weights,
    )


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
