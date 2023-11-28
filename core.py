import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from noises import transform_lambda, noise_baselines

class Smooth_Universal(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, noisegenerator1: torch.nn.Module,noisegenerator2: torch.nn.Module):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.noisegenerator1=noisegenerator1
        self.noisegenerator2 = noisegenerator2

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, noise_name: str,sigma: float) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        # mean,variance=self.noisegenerator(x)
        # mean=NoiseGenerator1(torch.ones(X.shape[0]).cuda().unsqueeze(1))
        mean=self.noisegenerator1(torch.ones(x.shape[0]).cuda().unsqueeze(1))
        # variance = self.noisegenerator2(torch.ones(x.shape[0]).cuda().unsqueeze(1)) * 5  # CIFAR10
        # variance=self.noisegenerator2(torch.ones(x.shape[0]).cuda().unsqueeze(1))*2 #imagenet
        variance = self.noisegenerator2(torch.ones(x.shape[0]).cuda().unsqueeze(1)) * 3  # mnist

        variance=torch.abs(variance)
        # mean=mean*0 #change
        # variance=sigma*variance
        # variance=sigma*torch.ones_like(variance).cuda() #change
        counts_selection = self._sample_noise(x, n0, batch_size,mean,variance,sigma,noise_name)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size,mean,variance,sigma,noise_name)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth_Universal.ABSTAIN, pABar,torch.sum(torch.log(torch.abs(variance))).cpu().data.numpy()
        else:
            # sigma_min=torch.min(torch.abs(variance)).cpu().data.numpy()
            # radius =  sigma_min* norm.ppf(pABar)`
            return cAHat, pABar,torch.sum(torch.log(torch.abs(variance))).cpu().data.numpy()


    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth_Universal.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, mean: torch.tensor, variance: torch.tensor, sigma, noise_name) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                mean_batch = mean.repeat((this_batch_size, 1, 1, 1)).reshape(batch.shape)
                variance_batch = variance.repeat((this_batch_size, 1, 1, 1)).reshape(batch.shape)
                lambd = transform_lambda(noise_name, sigma)
                noise = noise_baselines(noise_name, batch, lambd=lambd)-batch

                noise_input = batch+mean_batch+noise * variance_batch
                predictions = self.base_classifier(noise_input).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

class Smooth_Personalized(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, noisegenerator1: torch.nn.Module,noisegenerator2: torch.nn.Module):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.noisegenerator1=noisegenerator1
        self.noisegenerator2 = noisegenerator2

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, noise_name: str,sigma: float) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        # mean,variance=self.noisegenerator(x)
        # mean=NoiseGenerator1(torch.ones(X.shape[0]).cuda().unsqueeze(1))
        lambd = transform_lambda(noise_name, sigma)
        mean=self.noisegenerator1(x)*lambd
        # mean=self.noisegenerator1(x)
        variance=self.noisegenerator2(x+mean)
        variance=torch.abs(variance)
        # mean=mean*0 #change
        # variance=sigma*variance
        # variance=sigma*torch.ones_like(variance).cuda() #change
        counts_selection = self._sample_noise(x, n0, batch_size,mean,variance,sigma,noise_name)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size,mean,variance,sigma,noise_name)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth_Universal.ABSTAIN, pABar,variance
        else:
            # sigma_min=torch.min(torch.abs(variance)).cpu().data.numpy()
            # radius =  sigma_min* norm.ppf(pABar)`
            return cAHat, pABar,variance


    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth_Universal.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, mean: torch.tensor, variance: torch.tensor, sigma, noise_name) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                mean_batch = mean.repeat((this_batch_size, 1, 1, 1)).reshape(batch.shape)
                variance_batch = variance.repeat((this_batch_size, 1, 1, 1)).reshape(batch.shape)
                lambd = transform_lambda(noise_name, sigma)
                noise = noise_baselines(noise_name, batch, lambd=lambd)-batch

                noise_input = batch+mean_batch+noise*variance_batch
                predictions = self.base_classifier(noise_input).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

class Smooth_Preassigned(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int,pattern:torch.tensor):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.pattern=pattern
        # self.noisegenerator1=noisegenerator1
        # self.noisegenerator2 = noisegenerator2

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, noise_name: str,sigma: float) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        # mean,variance=self.noisegenerator(x)
        # mean=NoiseGenerator1(torch.ones(X.shape[0]).cuda().unsqueeze(1))
        # mean=self.noisegenerator1(x)
        # variance=self.noisegenerator2(x+mean)
        # variance=torch.abs(variance)
        # mean=mean*0 #change
        # variance=sigma*variance
        # variance=sigma*torch.ones_like(variance).cuda() #change
        counts_selection = self._sample_noise(x, n0, batch_size,sigma,noise_name)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size,sigma,noise_name)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth_Universal.ABSTAIN, pABar,torch.sum(torch.log(torch.abs(self.pattern))).cpu().data.numpy()
        else:
            # sigma_min=torch.min(torch.abs(variance)).cpu().data.numpy()
            # radius =  sigma_min* norm.ppf(pABar)`
            return cAHat, pABar,torch.sum(torch.log(torch.abs(self.pattern))).cpu().data.numpy()


    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth_Universal.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, sigma, noise_name) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                # mean_batch = mean.repeat((this_batch_size, 1, 1, 1)).reshape(batch.shape)
                # variance_batch = variance.repeat((this_batch_size, 1, 1, 1)).reshape(batch.shape)
                lambd = transform_lambda(noise_name, sigma)
                noise = noise_baselines(noise_name, batch, lambd=lambd)-batch

                noise_input = batch+noise * self.pattern.repeat((batch.shape[0],1,1,1)).reshape(batch.shape).float()
                predictions = self.base_classifier(noise_input).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]