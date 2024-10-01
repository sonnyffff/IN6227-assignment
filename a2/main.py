import BruteForce
import Apriori
import matplotlib.pyplot as plt


if __name__ == '__main__':
    transaction_sizes = list(range(0, 10000, 500))
    times = []  # List to store the execution times

    # Run the brute force algorithm for each transaction size
    for size in transaction_sizes:
        execution_time = BruteForce.run("bakery_sales_fixed_n.csv", 0, size)[0]
        times.append(execution_time)  # Store the time

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(transaction_sizes, times, marker='o')
    plt.title('Execution Time vs. Number of Transactions')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Execution Time (seconds)')
    plt.grid()
    plt.xticks(transaction_sizes, rotation=45)
    plt.tight_layout()
    plt.show()


    # unique_item_sizes = list(range(0, 11))
    # times = []  # List to store the execution times
    # uniques = []
    # # Run the brute force algorithm for each transaction size
    # for size in unique_item_sizes:
    #
    #     execution_time, add = BruteForce.run("bakery_sales_fixed_m.csv", size * 3, size * 3 + 2)
    #     times.append(execution_time)  # Store the time
    #     uniques.append(add)
    # print(uniques)
    # # Plotting the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(uniques, times, marker='o')
    # plt.title('Execution Time vs. Number of Unique Items')
    # plt.xlabel('Number of Unique Items')
    # plt.ylabel('Execution Time (seconds)')
    # plt.grid()
    # plt.xticks(uniques, rotation=45)
    # plt.tight_layout()
    # plt.show()


    # Apriori.run("bakery_sales_fixed_n.csv", 3, 3)
