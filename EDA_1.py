# ---------------------------------------- Exploratory Data Analysis Showcase ------------------------------------------

# Sales history of a pet shop that apparently only sells dogs

# Import stuff
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Create dataframe ----------

# Dog info
dogs = ["bulldog", "labrador", "german shephard", "poodle", "chihuahua", "boxer"]
colour = ["black", "brown", "ginger"]
weights = {
    "bulldog":         {"mu": 23.0, "sig": 3.0},
    "labrador":        {"mu": 33.0, "sig": 4.0},
    "german shephard": {"mu": 35.0, "sig": 6.2},
    "poodle":          {"mu": 30.0, "sig": 5.0},
    "chihuahua":       {"mu":  2.0, "sig": 0.5},
    "boxer":           {"mu": 29.0, "sig": 3.5}
}

# Initialize list of lists and populate with dog sales
list_of_lists = []

for sale in range(10000):
    # initialize inner list
    sale_list = []
    # dog breed
    dog_index = random.randint(0, 5)
    dog_breed = dogs[dog_index]
    sale_list.append(dog_breed)
    # dog price
    dog_price = int(random.randint(19, 199) * 0.7 * len(dogs[dog_index]))
    sale_list.append(dog_price)
    # dog colour
    if random.randint(1, 100) < 3:
        dog_colour = "albino"
    else:
        dog_colour = colour[random.randint(0,2)]
    sale_list.append(dog_colour)
    # dog weight
    mu = weights[dog_breed]["mu"]
    sig = weights[dog_breed]["sig"]
    dog_weight = round(np.random.normal(mu, sig),2)
    sale_list.append(dog_weight)
    # append sale list to list_of_lists
    list_of_lists.append(sale_list)

# convert to dataframe
dog_df = pd.DataFrame(list_of_lists, columns=["breed", "sale_price (£)", "colour", "weight (kg)"])

# print(dog_df)

# ---------- Questions ----------

# 1 - Display the sale volume and sum of sales money per dog breed, ordered by total money descending.
# 2 - What was the breed, colour and weight of the most expensive dog breed?
# 3 - Plot the average sale price for each dog colour as a bar chart.
# 4 - Plot a pie chart showing the % of each dog sold.
# 5 - Plot a histogram of the weights of all sold labradors. Do they appear to be normally distributed?
# 6 - Display the box and whisker plots for the weights of all dogs sold, side by side.
# 7 - Get sale info for any albino dogs
# 8 - Plot a scatter graph of dog weight against dog price. Work out correlation.

# ---------- Answers (uncomment selected) ----------

# # 1 - Display the sale volume and total income per dog breed, ordered by total income descending.
#
# print("Sale volume and total income per dog breed, ordered by total income descending.")
# print("")
# sale_vol_sales_per_dog = dog_df.groupby("breed").agg({"breed": "count","sale_price (£)": "sum"})
# sale_vol_sales_per_dog2 = sale_vol_sales_per_dog.rename(columns={"breed": "sale_vol",
#                                                                  "sale_price (£)": "total income"})
# print(sale_vol_sales_per_dog2.sort_values(by="total income", ascending=False))
# print("")
#
# # 2 - What was the breed, colour and weight of the most expensive dog breed?
#
# print("Breed, colour and weight of most expensive dog.")
# print("")
# print(dog_df[dog_df["sale_price (£)"] == dog_df["sale_price (£)"].max()])
# print("")
#
# # 3 - Plot the average sale price for each dog colour as a bar chart.
#
# print("Average sale price for each dog colour.")
# print("")
# avg_price_per_breed = dog_df.groupby("colour", as_index=False).agg({"sale_price (£)": "mean"})
# avg_price_per_breed2 = avg_price_per_breed.rename(columns={"sale_price (£)": "avg sale price"})
# print(avg_price_per_breed2)
# print("")
#
# plt.bar(avg_price_per_breed2["colour"], avg_price_per_breed2["avg sale price"])
# plt.xlabel("Dog Colour")
# plt.ylabel("Avg Sale Price (£)")
# plt.title("Avg sale price for per dog colour")
# plt.show()
#
# # 4 - Plot a pie chart showing the % of each dog sold.
#
# print("Percentage of each dog breed sold.")
# print("")
# perc_breed_sold = dog_df.groupby("breed", as_index=False).agg({"colour": "count"})
# perc_breed_sold2 = perc_breed_sold.rename(columns={"colour": "sale_vol"})
# perc_breed_sold2["total_sold"] = perc_breed_sold2["sale_vol"].sum()
# perc_breed_sold2["sold_perc"] = round(perc_breed_sold2["sale_vol"] * 100.0 / perc_breed_sold2["total_sold"],2)
# print(perc_breed_sold2)
# print("")
#
# plt.pie(perc_breed_sold2["sold_perc"], labels=perc_breed_sold2["sold_perc"])
# plt.title("% of each dog breed sold")
# plt.legend(perc_breed_sold2["breed"])
# plt.show()
#
# # 5 - Plot a histogram of the weights of all sold labradors. Do they appear to be normally distributed?
#
# # (At higher saler volumes, the result is closer to normal distribution.)
#
# print("Weights of labradors")
# print("")
# lab_weights = dog_df[["breed","weight (kg)"]]
# lab_weights = lab_weights[lab_weights["breed"] == "labrador"]
# lab_count = min(100, lab_weights["breed"].count())
# print(lab_weights)
#
# plt.hist(lab_weights["weight (kg)"], bins=lab_count, density=False)
# plt.xlabel("Weight (kg)")
# plt.ylabel("Sale_vol")
# plt.title("Weights of labradors")
# plt.show()
#
# # 6 - Display the box and whisker plots for the weights of all dogs sold, side by side.
#
# sns.set()
# sns.boxplot(x="breed", y="weight (kg)", data=dog_df)
# plt.xlabel("Dog Breed")
# plt.ylabel("Weight (kg)")
# plt.title("Weights of dog breeds.")
# plt.show()
#
# # 7 - Get sale info for any albino dogs
#
# print("Albino sales.")
# print("")
# print(dog_df[dog_df["colour"] == "albino"])
# print("")
#
# # 8 - Plot a scatter graph of dog weight against dog price. Then work out the Person correlation coefficient.
#
# plt.scatter(dog_df["weight (kg)"], dog_df["sale_price (£)"])
# plt.xlabel("Weight (kg)")
# plt.ylabel("Sale_price (£)")
# plt.title("Dog weight against dog price")
#
# corr_mat = np.corrcoef(dog_df["weight (kg)"], dog_df["sale_price (£)"])
# pearson_r = corr_mat[0,1]
# print("The Person correlation coefficient is: " + str(pearson_r))
#
# plt.show()
