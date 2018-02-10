using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Math.Statistics
{
    public class InformationGain
    {
        public static double EntropyShannon(double [] values)
        {

            Dictionary<double, long> freqs =
                Frequency(values);

            //Compute Entropy for probability
            return EntropyShannon(freqs);
        }

        /// <summary>
        /// Shannon Entropy with provided Frequency
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double EntropyShannon(Dictionary<double, long> freqs)
        {
            double entropy = 0;
            long sum = 0;
            
            //Compute Entropy for probability
            foreach (int value in freqs.Values)
                sum += value;

            double p;
            foreach (int value in freqs.Values)
            {
                p = (value + 0d) / sum;
                entropy += (0 - p * Logarithm.Log2(p));

            }
            return entropy;
        }


        /// <summary>
        /// Computes frequency of elements
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static Dictionary<double,long> 
            Frequency(double[] values)
        {
            Dictionary<double, long> freq =
                new Dictionary<double, long>();
            
            //Key is value, value is freq
            foreach (double value in values)
            {
                if (freq.Keys.Contains<double>(value))
                    freq[value] += 1;
                else
                    freq.Add(value, 1);
            }
            return freq;
        }
    }
}
