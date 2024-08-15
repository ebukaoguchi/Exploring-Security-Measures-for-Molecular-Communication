import matplotlib.pyplot as plt

import numpy as np


# Functions for forward and backward steps
def forward_step(channel_matrix, p_x):
    p_y_given_x = np.dot(channel_matrix.T, p_x)
    return p_y_given_x / p_y_given_x.sum()


def backward_step(channel_matrix, p_y_given_x):
    return np.dot(channel_matrix, p_y_given_x) / np.sum(np.dot(channel_matrix, p_y_given_x))


# Function to calculate channel capacity
def calculate_channel_capacity(channel_matrix, p_x, p_y_given_x):
    return np.sum(np.multiply(channel_matrix, np.log2(channel_matrix / np.outer(p_x, p_y_given_x))))


# Function to calculate secrecy capacity
def calculate_secrecy_capacity(channel_matrix, p_x, p_y_given_x):
    p_x_given_y = channel_matrix * np.outer(p_x, p_y_given_x) / np.sum(np.outer(p_x, p_y_given_x) * channel_matrix,
                                                                       axis=0)
    return np.sum(np.multiply(p_x_given_y, np.log2(p_x_given_y / np.outer(p_x, p_y_given_x))))


# Function to calculate mutual information
# Define function to calculate mutual information
def calculate_mutual_information(p_x, p_y_given_x):
    return np.sum(
        np.multiply(p_x, np.sum(np.multiply(p_y_given_x, np.log2(p_y_given_x / np.outer(p_x, p_y_given_x))), axis=1)))


# Adjust probabilities in the channel matrix based on release rate, distance, and environmental conditions
def adjust_channel_matrix(channel_matrix, release_rate, distance, environmental_conditions):
    # Adjust probabilities based on release rate
    channel_matrix *= release_rate

    # Adjust probabilities based on distance
    attenuation_factor = 1 / distance
    channel_matrix *= attenuation_factor

    # Adjust probabilities based on environmental conditions affecting diffusion rates
    diffusion_rate_factor = environmental_conditions
    channel_matrix *= diffusion_rate_factor

    return channel_matrix


# Blahut-Arimoto algorithm for channel capacity
def molecular_channel_capacity(channel_matrix, release_rate, distance, environmental_conditions, max_iterations=100,
                               epsilon=1e-6):
    num_input_symbols = channel_matrix.shape[0]
    p_x = np.ones(num_input_symbols) / num_input_symbols
    print("Input p_x: ", p_x)
    for _ in range(max_iterations):
        p_y_given_x = forward_step(channel_matrix, p_x)
        new_p_x = backward_step(channel_matrix, p_y_given_x)

        if np.linalg.norm(new_p_x - p_x.mean()) < epsilon:
            break

        p_x = new_p_x

    return p_x


# Parameters for communication channels
channel_matrix_alice_bob = np.array([[0.4, 0.1], [0.6, 0.4]])
print("channel_matrix_alice_bob: ", channel_matrix_alice_bob)
channel_matrix_alice_eve = np.array([[0.2, 0.05], [0.3, 0.2]])
print("channel_matrix_alice_eve: ", channel_matrix_alice_eve)

# Parameters for transmission
release_rate_alice_bob = 1.5
distance_alice_bob = 2.0
environmental_conditions_alice_bob = 1.2

# Adjust channel matrices
adjusted_channel_matrix_alice_bob = adjust_channel_matrix(channel_matrix_alice_bob, release_rate_alice_bob,
                                                          distance_alice_bob, environmental_conditions_alice_bob)
print("adjusted_channel_matrix_alice_bob", adjusted_channel_matrix_alice_bob)
adjusted_channel_matrix_alice_eve = adjust_channel_matrix(channel_matrix_alice_eve, release_rate_alice_bob,
                                                          distance_alice_bob, environmental_conditions_alice_bob)
print("adjusted_channel_matrix_alice_eve", adjusted_channel_matrix_alice_eve)

# Calculate channel capacity for Alice to Bob
optimized_input_distribution_alice_bob = molecular_channel_capacity(adjusted_channel_matrix_alice_bob,
                                                                    release_rate_alice_bob, distance_alice_bob,
                                                                    environmental_conditions_alice_bob)
print("optimized_input_distribution_alice_bob", optimized_input_distribution_alice_bob)
p_y_given_x_alice_bob = forward_step(adjusted_channel_matrix_alice_bob, optimized_input_distribution_alice_bob)
print("p_y_given_x_alice_bob: ", p_y_given_x_alice_bob)
channel_capacity_alice_bob = calculate_channel_capacity(adjusted_channel_matrix_alice_bob,
                                                        optimized_input_distribution_alice_bob, p_y_given_x_alice_bob)
print("channel_capacity_alice_bob", channel_capacity_alice_bob)

# Calculate channel capacity for Alice to Eve
optimized_input_distribution_alice_eve = molecular_channel_capacity(adjusted_channel_matrix_alice_eve,
                                                                    release_rate_alice_bob, distance_alice_bob,
                                                                    environmental_conditions_alice_bob)
print("optimized_input_distribution_alice_eve", optimized_input_distribution_alice_eve)
p_y_given_x_alice_eve = forward_step(adjusted_channel_matrix_alice_eve, optimized_input_distribution_alice_eve)
print("p_y_given_x_alice_eve: ", p_y_given_x_alice_eve)
channel_capacity_alice_eve = calculate_channel_capacity(adjusted_channel_matrix_alice_eve,
                                                        optimized_input_distribution_alice_eve, p_y_given_x_alice_eve)
print("channel_capacity_alice_eve", channel_capacity_alice_eve)

# Calculate secrecy capacity
secrecy_capacity_alice_bob = calculate_secrecy_capacity(adjusted_channel_matrix_alice_bob,
                                                        optimized_input_distribution_alice_bob, p_y_given_x_alice_bob)

# Calculate mutual information
mutual_information_alice_bob = calculate_mutual_information(optimized_input_distribution_alice_bob,
                                                            p_y_given_x_alice_bob)
mutual_information_alice_eve = calculate_mutual_information(optimized_input_distribution_alice_eve,
                                                            p_y_given_x_alice_eve)

# Define the Hamming code generator matrix
G = np.array([[1, 0, 0, 0, 1, 1, 0],
              [0, 1, 0, 0, 1, 0, 1],
              [0, 0, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1]])

# Define the Hamming code parity check matrix
H = np.array([[1, 1, 0, 1, 1, 0, 0],
              [1, 0, 1, 1, 0, 1, 0],
              [0, 1, 1, 1, 0, 0, 1]])


# Function to encode a message using Hamming code
def hamming_encode(message):
    message = np.array(message)
    # Calculate the codeword by multiplying the message with the generator matrix
    codeword = np.dot(message, G) % 2
    return codeword


# Function to decode a received message using Hamming code
def hamming_decode(received_codeword):
    received_codeword = np.array(received_codeword)
    # Calculate the syndrome by multiplying the received codeword with the parity check matrix
    syndrome = np.dot(received_codeword, H.T) % 2
    # If syndrome is non-zero, there's an error in the received codeword
    if np.any(syndrome):
        # Find the index of the error bit and flip it
        error_index = np.sum(syndrome * 2 ** np.arange(len(syndrome)))
        received_codeword[error_index] ^= 1
    # Extract the original message from the received codeword
    original_message = received_codeword[:4]
    return original_message


# Example usage:
message_alice_bob = [1, 0, 1, 1]
print("Original Message from Alice to Bob:", message_alice_bob)

# Encode the message using Hamming code
encoded_message_alice_bob = hamming_encode(message_alice_bob)
print("Encoded Message from Alice to Bob:", encoded_message_alice_bob)

# Simulate transmission through a noisy channel (in this example, Eve flips a bit)
flipped_encoded_message = np.array(encoded_message_alice_bob)
flipped_encoded_message[6] ^= 1  # Flipping the 7th bit

print("Original Message from Alice to Bob:", message_alice_bob)
print("Encoded Message from Alice to Bob:", encoded_message_alice_bob)
print("Eve flipped bit at index:", 6)

# Decode the received codeword using Hamming code
decoded_message_bob = hamming_decode(flipped_encoded_message)
print("Decoded Message at Bob:", decoded_message_bob)

# Simulate transmission through a noisy channel (in this example, Eve flipped two bits)
received_codeword_alice_bob = encoded_message_alice_bob.copy()
received_codeword_alice_bob[2] ^= 1  # Flip one bit
received_codeword_alice_bob[6] ^= 1  # Flip another bit

# Decode the received codeword using Hamming code
decoded_message_alice_bob = hamming_decode(received_codeword_alice_bob)
print("Decoded Message at Bob1:", decoded_message_alice_bob)

# Results
print("Results for Legitimate Channel (Alice to Bob):")
print("Channel Capacity (Alice to Bob):", channel_capacity_alice_bob)
print("Secrecy Capacity :", secrecy_capacity_alice_bob)
print("Mutual Information (Alice to Bob):", mutual_information_alice_bob)

print("\nResults for Eavesdropped Channel (Alice to Eve):")
print("Channel Capacity (Alice to Eve):", channel_capacity_alice_eve)
# print("Secrecy Capacity (Alice to Eve):", secrecy_capacity_alice_eve)
print("Mutual Information (Alice to Eve):", mutual_information_alice_eve)

# Parameters for transmission. Release rate and environmental condition is kept constant while vary the distance
release_rate_alice_bob = 1.5
environmental_conditions_alice_bob = 1.2

# Define distances
distances = np.linspace(1, 3, 3)

# Initialize lists to store results
channel_capacities_alice_bob = []
channel_capacities_alice_eve = []

# Calculate channel capacities for different distances
for distance in distances:
    adjusted_channel_matrix_alice_bob = adjust_channel_matrix(channel_matrix_alice_bob, release_rate_alice_bob,
                                                              distance, environmental_conditions_alice_bob)
    adjusted_channel_matrix_alice_eve = adjust_channel_matrix(channel_matrix_alice_eve, release_rate_alice_bob,
                                                              distance, environmental_conditions_alice_bob)

    optimized_input_distribution_alice_bob = molecular_channel_capacity(adjusted_channel_matrix_alice_bob,
                                                                        release_rate_alice_bob, distance,
                                                                        environmental_conditions_alice_bob)
    p_y_given_x_alice_bob = forward_step(adjusted_channel_matrix_alice_bob, optimized_input_distribution_alice_bob)
    channel_capacity_alice_bob = calculate_channel_capacity(adjusted_channel_matrix_alice_bob,
                                                            optimized_input_distribution_alice_bob,
                                                            p_y_given_x_alice_bob)
    channel_capacities_alice_bob.append(channel_capacity_alice_bob)

    optimized_input_distribution_alice_eve = molecular_channel_capacity(adjusted_channel_matrix_alice_eve,
                                                                        release_rate_alice_bob, distance,
                                                                        environmental_conditions_alice_bob)
    p_y_given_x_alice_eve = forward_step(adjusted_channel_matrix_alice_eve, optimized_input_distribution_alice_eve)
    channel_capacity_alice_eve = calculate_channel_capacity(adjusted_channel_matrix_alice_eve,
                                                            optimized_input_distribution_alice_eve,
                                                            p_y_given_x_alice_eve)
    channel_capacities_alice_eve.append(channel_capacity_alice_eve)

# Plot results
plt.figure()
# plt.figure(figsize=(3.5, 2.5))  # Width: 3.5 inches, Height: 2.5 inches
plt.plot(distances, channel_capacities_alice_bob, label='Alice to Bob')
plt.plot(distances, channel_capacities_alice_eve, label='Alice to Eve')
plt.xlabel('Distance')
plt.ylabel('Channel Capacity')
plt.title('Channel Capacity vs. Distance')
plt.legend()
plt.grid(True)
# plt.savefig('channel_capacity_vs_distance.png')
# Save the plot to the specified location as .eps file
# uncomment if you want to generate plots and save
# plt.savefig('/home/cse-nghose-23/Desktop/channel_capacity_vs_distance.eps', format='eps', bbox_inches='tight')
# plt.savefig('/home/cse-nghose-23/Desktop/channel_capacity_vs_distance.png')
# plt.show()


# Parameters for transmission. distance and environmental condition is kept constant while vary the release rate
distance = 2.0
environmental_conditions = 1.2

# Define release rates
# release_rates = np.linspace(0.5, 2, 10)
release_rates = np.linspace(1, 3, 3)

# Initialize lists to store results
channel_capacities_alice_bob = []
channel_capacities_alice_eve = []

# Calculate channel capacities for different release rates
for release_rate in release_rates:
    adjusted_channel_matrix_alice_bob = adjust_channel_matrix(channel_matrix_alice_bob, release_rate, distance,
                                                              environmental_conditions)
    optimized_input_distribution_alice_bob = molecular_channel_capacity(adjusted_channel_matrix_alice_bob, release_rate,
                                                                        distance, environmental_conditions)
    p_y_given_x_alice_bob = forward_step(adjusted_channel_matrix_alice_bob, optimized_input_distribution_alice_bob)
    channel_capacity_alice_bob = calculate_channel_capacity(adjusted_channel_matrix_alice_bob,
                                                            optimized_input_distribution_alice_bob,
                                                            p_y_given_x_alice_bob)
    channel_capacities_alice_bob.append(channel_capacity_alice_bob)

    adjusted_channel_matrix_alice_eve = adjust_channel_matrix(channel_matrix_alice_eve, release_rate, distance,
                                                              environmental_conditions)
    optimized_input_distribution_alice_eve = molecular_channel_capacity(adjusted_channel_matrix_alice_eve, release_rate,
                                                                        distance, environmental_conditions)
    p_y_given_x_alice_eve = forward_step(adjusted_channel_matrix_alice_eve, optimized_input_distribution_alice_eve)
    channel_capacity_alice_eve = calculate_channel_capacity(adjusted_channel_matrix_alice_eve,
                                                            optimized_input_distribution_alice_eve,
                                                            p_y_given_x_alice_eve)
    channel_capacities_alice_eve.append(channel_capacity_alice_eve)

plt.figure()
# Plot results
plt.plot(release_rates, channel_capacities_alice_bob, label='Alice to Bob')
plt.plot(release_rates, channel_capacities_alice_eve, label='Alice to Eve')
plt.xlabel('Release Rate')
plt.ylabel('Channel Capacity')
plt.title('Channel Capacity vs. Release Rate')
plt.legend()
plt.grid(True)
# uncomment if you want to generate plots and save
# plt.savefig('/home/cse-nghose-23/Desktop/channel_capacity_vs_release_rate.eps', format='eps', bbox_inches='tight')
# plt.show


# Parameters for transmission. Release rate and environmental condition is kept constant while vary the environmental condition
release_rate = 1.5
distance = 2.0

# Define environmental conditions
# environmental_conditions = np.linspace(0.5, 2, 10)
environmental_conditions = np.linspace(1, 3, 3)

# Initialize lists to store results
channel_capacities_alice_bob = []
channel_capacities_alice_eve = []

# Calculate channel capacities for different environmental conditions
for env_condition in environmental_conditions:
    adjusted_channel_matrix_alice_bob = adjust_channel_matrix(channel_matrix_alice_bob, release_rate, distance,
                                                              env_condition)
    optimized_input_distribution_alice_bob = molecular_channel_capacity(adjusted_channel_matrix_alice_bob, release_rate,
                                                                        distance, env_condition)
    p_y_given_x_alice_bob = forward_step(adjusted_channel_matrix_alice_bob, optimized_input_distribution_alice_bob)
    channel_capacity_alice_bob = calculate_channel_capacity(adjusted_channel_matrix_alice_bob,
                                                            optimized_input_distribution_alice_bob,
                                                            p_y_given_x_alice_bob)
    channel_capacities_alice_bob.append(channel_capacity_alice_bob)

    adjusted_channel_matrix_alice_eve = adjust_channel_matrix(channel_matrix_alice_eve, release_rate, distance,
                                                              env_condition)
    optimized_input_distribution_alice_eve = molecular_channel_capacity(adjusted_channel_matrix_alice_eve, release_rate,
                                                                        distance, env_condition)
    p_y_given_x_alice_eve = forward_step(adjusted_channel_matrix_alice_eve, optimized_input_distribution_alice_eve)
    channel_capacity_alice_eve = calculate_channel_capacity(adjusted_channel_matrix_alice_eve,
                                                            optimized_input_distribution_alice_eve,
                                                            p_y_given_x_alice_eve)
    channel_capacities_alice_eve.append(channel_capacity_alice_eve)

plt.figure()
# Plot results
plt.plot(environmental_conditions, channel_capacities_alice_bob, label='Alice to Bob')
plt.plot(environmental_conditions, channel_capacities_alice_eve, label='Alice to Eve')
plt.xlabel('Environmental Conditions')
plt.ylabel('Channel Capacity')
plt.title('Channel Capacity vs. Environmental Conditions')
plt.legend()
plt.grid(True)
# uncomment if you want to generate plots and save
# plt.savefig('/home/cse-nghose-23/Desktop/channel_capacity_vs_environmental_conditions.eps', format='eps', bbox_inches='tight')
# plt.show()

# Parameters for transmission
release_rate_alice_bob = 1.5
environmental_conditions_alice_bob = 1.2

# Define distances
distances = np.linspace(1, 3, 10)

# Initialize lists to store results
secrecy_capacities = []

# Calculate secrecy capacity for different distances
for distance in distances:
    adjusted_channel_matrix_alice_bob = adjust_channel_matrix(channel_matrix_alice_bob, release_rate_alice_bob,
                                                              distance, environmental_conditions_alice_bob)
    optimized_input_distribution_alice_bob = molecular_channel_capacity(adjusted_channel_matrix_alice_bob,
                                                                        release_rate_alice_bob, distance,
                                                                        environmental_conditions_alice_bob)
    p_y_given_x_alice_bob = forward_step(adjusted_channel_matrix_alice_bob, optimized_input_distribution_alice_bob)
    secrecy_capacity_alice_bob = calculate_secrecy_capacity(adjusted_channel_matrix_alice_bob,
                                                            optimized_input_distribution_alice_bob,
                                                            p_y_given_x_alice_bob)
    secrecy_capacities.append(secrecy_capacity_alice_bob)

# Plot results
plt.plot(distances, secrecy_capacities)
plt.xlabel('Distance')
plt.ylabel('Secrecy Capacity')
plt.title('Secrecy Capacity vs. Distance')
plt.grid(True)
# uncomment if you want to generate plots and save
# plt.savefig('/home/cse-nghose-23/Desktop/secrecy_capacity_vs_distance.eps', format='eps', bbox_inches='tight')
# plt.show()