using System;


namespace Dasmic.MLLib.Math.Statistics
{
    public class DistributionNormal
    {
        protected double _largeTolerance; //Higher tolerance
        protected double _largeStep; //Higher tolerance
        protected double _smallStep;


        public DistributionNormal()
        {
            _largeTolerance = .01; //Higher tolerance
            _largeStep = .05; //Higher tolerance
            _smallStep = .001;
        }

        /// <summary>
        /// Use Integration by parts technique:
        /// 
        /// Source:
        /// https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double CumulativeDistributionFunction(double value, int seriesMax)
        {
            
            double tmpValue=value; //Already done for index 0
            double seriesSum = tmpValue;
            //Start from 1 
            for (int ii=1;ii<seriesMax; ii++)
            {
                tmpValue = (tmpValue * value * value) / 
                    (2 * ii + 1);
                seriesSum += tmpValue;

            }
            
            return 0.5 + (seriesSum / System.Math.Sqrt(2 * System.Math.PI)) *
                                System.Math.Exp(-(value * value) / 2);
            
        }

        /// <summary>
        /// Retutns the CDF with seriesMax default to 100
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double CumulativeDistributionFunction(double value)
        {
            return CumulativeDistributionFunction(value, 100);
        }

        
        /// <summary>
        /// Returns the PDF of a given value based on SD and Mean
        /// </summary>
        /// <param name="value"></param>
        /// <param name="mean"></param>
        /// <param name="standardDeviation"></param>
        /// <returns></returns>
        public double ProbabilityDensityFunction(double value,
                        double mean, double standardDeviation)
        {
            double tmp1 = 1.0 / (System.Math.Pow(2 * System.Math.PI, 0.5) * standardDeviation);
            double tmp2 = System.Math.Pow(value - mean, 2.0)/(2 * standardDeviation * standardDeviation) ;
            tmp2 = System.Math.Exp(0 - tmp2);
            return tmp1 * tmp2;
        }

        /// <summary>
        /// Returns the ZValue for a cofidence internal
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double ZValue_ConfidenceInterval(
                                double confidencePercent,
                                double tolerance=.0001)
        {
            //Go about the mean
            //Mean = 0
            double confidenceBase1 = confidencePercent / 100;
            double halfConfidence = confidencePercent / 2.0;

            return ZValue(halfConfidence, tolerance);

        }


        /// <summary>
        /// Returns the ZValue for a positive side
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double ZValue(double percent,
                                double tolerance=.0001)
        {

            //Go about the mean
            //Mean = 0
            double halfConfidence = percent / 100.0;
            double value = 0.0;
            double offsetMean = _largeStep;
            double areaMean = CumulativeDistributionFunction(0); //Area left of the mean

            //Higher steps
            value = CumulativeDistributionFunction(offsetMean) - areaMean;
            while (System.Math.Abs(value - halfConfidence) > _largeTolerance)
            {
                offsetMean += _largeStep;
                value = CumulativeDistributionFunction(offsetMean) - areaMean;
            }

            //Make sure value is more since we decrease in next step
            while (value <= halfConfidence)
            {
                offsetMean += _largeStep;
                value = CumulativeDistributionFunction(offsetMean) - areaMean;
            }

            //Now execute small steps - refine value
            double prevDiff = 0; 
            double diff = System.Math.Abs(value - halfConfidence);
            while (System.Math.Abs(diff) > tolerance)
            {
                offsetMean -= _smallStep;
                value = CumulativeDistributionFunction(offsetMean) - areaMean;
                prevDiff = diff;
                diff = System.Math.Abs(value - halfConfidence);

                if(diff > prevDiff)
                { //We are diverging now, hence break
                    break;
                }
            }

            return offsetMean; //This is Zvalue since SD=1
        }

        


        //This can be removed
        // ACM Algorithm #209
        public double Gauss(double z)
        {
            // input = z-value (-inf to +inf)
            // output = p under Normal curve from -inf to z
            // e.g., if z = 0.0, function returns 0.5000
            // ACM Algorithm #209
            double y; // 209 scratch variable
            double p; // result. called ‘z’ in 209
            double w; // 209 scratch variable

            if (z == 0.0)
                p = 0.0;
            else
            {
                y = System.Math.Abs(z) / 2;
                if (y >= 3.0)
                {
                    p = 1.0;
                }
                else if (y < 1.0)
                {
                    w = y * y;
                    p = ((((((((0.000124818987 * w
                       - 0.001075204047) *w + 0.005198775019) *w
                    - 0.019198292004) *w + 0.059054035642) *w
                    - 0.151968751364) *w + 0.319152932694) *w
                    - 0.531923007300) *w + 0.797884560593) *y * 2.0;
                }
                else
                {
                    y = y - 2.0;
                    p = (((((((((((((-0.000045255659 * y
                      + 0.000152529290) * y - 0.000019538132) *y
                    - 0.000676904986) *y + 0.001390604284) *y
                    - 0.000794620820) *y - 0.002034254874) *y
                   + 0.006549791214) *y - 0.010557625006) *y
                  + 0.011630447319) *y - 0.009279453341) *y
                 + 0.005353579108) *y - 0.002141268741) *y
                + 0.000535310849) *y + 0.999936657524;
                }
            }

            if (z > 0.0)
                return (p + 1.0) / 2;
            else
                return (1.0 - p) / 2;
        } // Gauss()

    }
}
