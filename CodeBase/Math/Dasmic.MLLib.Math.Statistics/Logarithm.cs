using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Math.Statistics
{
    public class Logarithm
    {
        /// <summary>
        /// Returns logarithm base 2 of a given number
        /// </summary>
        /// <param name="number"></param>
        public static double Log2(double number)
        {
            return System.Math.Log(number, 2);
        }
    }
}
