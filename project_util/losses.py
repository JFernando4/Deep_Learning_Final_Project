import tensorflow as tf


def compute_ftau(p, tau=1.0):
    f_tau = tf.nn.softmax(tf.div(p, tau))
    return f_tau


def compute_Ftau(p, tau=1.0):
    F_tau_sum = tf.reduce_sum(tf.exp(tf.scalar_mul(1.0/tau, p)), 1)
    F_tau = tf.scalar_mul(tau, tf.log(F_tau_sum))
    return F_tau


def compute_Ftau_star(p, tau=1.):
    first_term = tf.scalar_mul(tau, p)
    second_term = tf.log(p)
    Ftau_star = tf.reduce_sum(tf.multiply(first_term, second_term), 1)
    return Ftau_star


def expected_cost(z_hat, y, tau=1.0):
    c = tf.subtract(1.0, y) # 1 - y
    f_tau = compute_ftau(z_hat, tau)
    expected_c = tf.reduce_sum(tf.multiply(c, f_tau), 1)
    return expected_c


def kl_divergence_rl(z_hat, y, tau=1.0):
    F_tau = compute_Ftau(y, tau)
    f_tau = compute_ftau(z_hat, tau)
    middle_term = tf.negative(tf.reduce_sum(tf.multiply(y, f_tau), 1))
    F_tau_star = compute_Ftau_star(f_tau, tau)
    kl_divergence = tf.add_n([F_tau, middle_term, F_tau_star])
    return kl_divergence


def kl_divergence_ml(z_hat, y, tau=1.0):
    F_tau = compute_Ftau(z_hat, tau)
    f_tau = compute_ftau(y, tau)
    middle_term = tf.negative(tf.reduce_sum(tf.multiply(z_hat, f_tau), 1))
    F_tau_star = compute_Ftau_star(f_tau, tau)
    kl_divergence = tf.add_n([F_tau, middle_term, F_tau_star])
    return kl_divergence
