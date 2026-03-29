import torch


"""
Frequency Evaluations

# Example usage
input = torch.randint(-1024, 1024, size=(64, 60, 20)).float()
target = input - 1024 * torch.randn(64, 60, 20)
print(input, target)
avg_js_divergence, avg_bha_distance = evaluate_coefficients_restoration(input, target)
print(f"Average JS Divergence: {avg_js_divergence.item()}")
print(f"Average Bhattacharyya Distance: {avg_bha_distance.item()}")

"""


def kl_divergence(p, q, epsilon=1e-10):
    """
    Computes the Kullback-Leibler divergence between two probability distributions.
    :param epsilon: float, a small constant added to the probabilities to avoid division by zero.
    :return: torch.tensor, the KL divergence between p and q.
    """
    p, q = p + epsilon, q + epsilon
    return (p * (p / q).log()).sum()


def js_divergence(p, q):
    """
    Computes the Jensen-Shannon divergence between two probability distributions.
    :return: torch.tensor, the JS divergence between p and q.
    """
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


def bhattacharyya_distance(p, q):
    """
    Computes the Bhattacharyya distance between two probability distributions.
    :return: torch.tensor, the Bhattacharyya distance between p and q.
    """
    return -torch.log((p.sqrt() * q.sqrt()).sum())


def dct_coefficients_to_probability_distribution(dct_coeffs, num_bin=64, minmax='1024'):
    """
    Converts DCT coefficients to a probability distribution.
    dct_coeffs: torch.tensor, the DCT coefficients.
    return: torch.tensor, the probability distribution derived from the DCT coefficients.
    """
    assert minmax in ['1024', 'auto']

    # Compute the histogram of the DCT coefficients
    if minmax == 'auto':
        hist = torch.histc(dct_coeffs, bins=num_bin, min=dct_coeffs.min(), max=dct_coeffs.max())
    else:
        hist = torch.histc(dct_coeffs, bins=num_bin, min=-1024, max=1024)

    # Normalize the histogram to form a probability distribution
    prob_dist = hist / hist.sum()

    return prob_dist


def evaluate_coefficients_restoration(input_dct_coeffs, target_dct_coeffs):
    """
    :param: torch.tensor, the DCT coefficients in 64xHxW
    """
    # Ensure that the input and target tensors have the same shape

    if input_dct_coeffs.shape != target_dct_coeffs.shape:
        print("different in input dct coefficient shape: ", input_dct_coeffs.shape, target_dct_coeffs.shape)
    # assert input_dct_coeffs.shape == target_dct_coeffs.shape

    # Initialize JS divergence and Bha distance accumulators
    js_divergence_accumulator = 0
    bha_distance_accumulator = 0

    # Iterate over each frequency component
    for c in range(input_dct_coeffs.shape[0]):
        # Get the DCT coefficients for the current frequency component
        input_dct_component = input_dct_coeffs[c]
        target_dct_component = target_dct_coeffs[c]

        # Convert the DCT coefficients to probability distributions
        input_prob_dist = dct_coefficients_to_probability_distribution(input_dct_component)
        target_prob_dist = dct_coefficients_to_probability_distribution(target_dct_component)

        # Compute the JS divergence and Bha distance for the current frequency component
        js_div = js_divergence(input_prob_dist, target_prob_dist)
        bha_dis = bhattacharyya_distance(input_prob_dist, target_prob_dist)

        # print(f"channel:{c}, js:{js_div}, bha:{bha_dis}")

        # Accumulate the JS divergence and Bha distance
        js_divergence_accumulator += js_div
        bha_distance_accumulator += bha_dis

    # Compute the average JS divergence and Bha distance
    avg_js_divergence = js_divergence_accumulator / input_dct_coeffs.shape[0]
    avg_bha_distance = bha_distance_accumulator / input_dct_coeffs.shape[0]

    return avg_js_divergence, avg_bha_distance

