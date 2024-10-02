import BruteForce
import Apriori
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # transaction_sizes = list(range(0, 10000, 500))
    # brute_force_times = []  # List to store the execution times for brute-force
    # apriori_times = []
    # apriori_times2 = []
    #
    # # Run the brute force algorithm for each transaction size
    # for size in transaction_sizes:
    #     execution_time_bf = BruteForce.run("bakery_sales_fixed_n.csv", 0, size)[0]
    #     brute_force_times.append(execution_time_bf)  # Store the time
    #     execution_time_ap = Apriori.run("bakery_sales_fixed_n.csv", 0, size, 0.05, 0.5)
    #     apriori_times.append(execution_time_ap)
    #     # execution_time_ap2 = Apriori.run("bakery_sales_fixed_n11.csv", 0, size, 10, 10)
    #     # apriori_times2.append(execution_time_ap2)
    #     # execution_time_ap = Apriori.run("bakery_sales_fixed_n11.csv", 0, size, 0.5, 1)
    #     # apriori_times3.append(execution_time_ap)
    # # Plotting the results
    # plt.figure(figsize=(10, 6))
    # # plt.plot(transaction_sizes, times, marker='o')
    # plt.plot(transaction_sizes, brute_force_times, marker='o', label='Brute Force Runtime', color='b')
    # plt.plot(transaction_sizes, apriori_times, marker='x', linestyle='--', label='Apriori Runtime', color='r')
    # # plt.plot(transaction_sizes, apriori_times2, marker='x', linestyle='--', label='Apriori Runtime', color='y')
    #
    # plt.title('Execution Time vs. Number of Transactions')
    # plt.xlabel('Number of Transactions')
    # plt.ylabel('Execution Time (seconds)')
    # plt.grid()
    # plt.legend()
    # plt.xticks(transaction_sizes, rotation=45)
    # plt.tight_layout()
    # plt.show()


    unique_item_sizes = list(range(0, 11))
    brute_force_times = []  # List to store the execution times for brute-force
    apriori_times = []
    uniques = []
    # Run the brute force algorithm for each transaction size
    for size in unique_item_sizes:

        execution_time, add = BruteForce.run("bakery_sales_fixed_m4k.csv", size * 4000, size * 4000 + 3999)
        brute_force_times.append(execution_time)  # Store the time

        uniques.append(add)

        execution_time2 = Apriori.run("bakery_sales_fixed_m4k.csv", size * 4000, size * 4000 + 3999, 0.05, 0.5)
        apriori_times.append(execution_time2)
    # print(uniques)
    # Plotting the results
    plt.figure(figsize=(10, 6))
    # plt.plot(uniques, times, marker='o')
    plt.plot(uniques, brute_force_times, marker='o', label='Brute Force Runtime', color='b')
    plt.plot(uniques, apriori_times, marker='x', linestyle='--', label='Apriori Runtime', color='r')
    plt.title('Execution Time vs. Number of Unique Items')
    plt.xlabel('Number of Unique Items')
    plt.ylabel('Execution Time (seconds)')
    plt.grid()
    plt.xticks(uniques, rotation=45)
    plt.tight_layout()
    plt.show()



    # transaction_sizes = list(range(0, 10000, 500))
    # apriori_times = []
    #
    # # Run the brute force algorithm for each transaction size
    # for size in transaction_sizes:
    #     execution_time_ap = Apriori.run("bakery_sales_fixed_n.csv", 0, size, 0.05, 0.5)
    #     apriori_times.append(execution_time_ap)
    # # Plotting the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(transaction_sizes, apriori_times, marker='x', linestyle='--', label='Apriori Runtime', color='r')
    #
    # plt.title('Execution Time vs. Number of Transactions')
    # plt.xlabel('Number of Transactions')
    # plt.ylabel('Execution Time (seconds)')
    # plt.grid()
    # plt.legend()
    # plt.xticks(transaction_sizes, rotation=45)
    # plt.tight_layout()
    # plt.show()
