using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.DataManagement
{
    public static class Conversion
    {
        #region String/Double Conversion Functions

        /// <summary>
        /// Converts a passed 1D double array to string array
        /// </summary>
        /// <param name="origArray"></param>        
        /// <returns></returns>
        public static string[] ConvertToStringArray(double[] origArray)
        {
            string[] sArray = origArray.OfType<double>().Select(o => o.ToString()).ToArray();
            return sArray;
        }

        /// <summary>
        /// Replaces all values of a matching string value with a new string value
        /// Returns new array
        /// </summary>
        /// <param name="origArray"></param>
        /// <param name="value"></param>
        /// <param name="replaceWithValue"></param>
        /// <returns></returns>
        public static string[] ReplaceWithString1D(string[] origArray,string oldValue,string newValue)
        {         
            string[] sArray = origArray.Select(x => x.Replace(oldValue,newValue)).ToArray();
            return sArray;
        }

        #endregion String/Double Conversion Functions
    }
}
