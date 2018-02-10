using System.Collections.Generic;
using System.Threading.Tasks;


namespace Dasmic.Portable.Core
{
    public static class SupportFunctions
    {
        /// <summary>
        /// Return a boolean value whether value1 and value2 are same or not
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns></returns>
        public static bool DoubleCompare(double value1, double value2)
        {
            double tol = .01;
            if (value2 >= value1 - tol && value2 <= value1 + tol)
                return true;
            else
                return false;               
        }


        /// <summary>
        /// Get sorted values of KeyValuePair in ascending order
        /// </summary>
        /// <param name="allValues"></param>
        /// <returns></returns>
        public static List<KeyValuePair<int, double>>
                GetSortedKeyValuePair(KeyValuePair<int, double> [] allValues)
        {
            List<KeyValuePair<int, double>> sortedList =
                new List<KeyValuePair<int, double>>(allValues);

            sortedList.Sort(
                delegate (KeyValuePair<int, double> pair1,
                    KeyValuePair<int, double> pair2)
                {
                    return pair1.Value.CompareTo(pair2.Value);
                });
            return sortedList;
        }

        /// <summary>
        /// Returns a 1D for a specific row in a 2D array uptill specific column
        /// </summary>
        /// <param name="mainArray"></param>
        /// <param name="mainArrayRowIdx"></param>
        /// <param name="endColumnIdx"></param>
        /// <returns></returns>
        /*[System.Obsolete("Method is deprecated, use new method with long for mainArrayRowIdx")]
        public static double[] GetLinearArray(double[][] mainArray, int mainArrayRowIdx, int endColumnIdx)
        {
            double[] data = new double[endColumnIdx + 1];
            for (int col = 0; col <= endColumnIdx; col++)
            {
                data[col] = mainArray[col][mainArrayRowIdx];
            }
            return data;
        }*/

        /// <summary>
        /// Returns a 1D for a specific row in a 2D array uptill specific column
        /// </summary>
        /// <param name="mainArray"></param>
        /// <param name="mainArrayRowIdx"></param>
        /// <param name="endColumnIdx"></param>
        /// <returns></returns>
        public static double[] GetLinearArray(double[][] mainArray, 
                                    long mainArrayRowIdx, int endColumnIdx)
        {
            double[] data = new double[endColumnIdx + 1];
            for (int col = 0; col <= endColumnIdx; col++)
            {
                data[col] = mainArray[col][mainArrayRowIdx];
            }
            return data;
        }

        /// <summary>
        /// Returns a 1D for a specific row in a 2D array for all columns
        /// </summary>
        /// <param name="mainArray"></param>
        /// <param name="mainArrayRowIdx"></param>
        /// <returns></returns>
        public static double[] GetLinearArray(double[][] mainArray, long mainArrayRowIdx)
        {
            return GetLinearArray(mainArray, mainArrayRowIdx, mainArray.Length - 1);
        }

        /// <summary>
        /// Set up a 2D array and initializes it
        /// </summary>
        /// <param name="cols"></param>
        /// <param name="rows"></param>
        /// <param name="maxThreads"></param>
        /// <returns></returns>
        public static double[][] 
            Get2DArray(int cols, int rows,int maxThreads=-1)
        {
            double[][] newArray = new double[cols][];
            for (int col = 0; col < cols; col++) //Do no parallize here, parallelize the value assignment
            {
                newArray[col] = new double[rows];
                InitArray(newArray[col],maxThreads);
            }
            return newArray;
        }

        /// <summary>
        /// Initializes an array with 0
        /// </summary>
        /// <param name="array"></param>
        /// <param name="maxThreads"></param>
        public static void InitArray(double[] array, int maxThreads)
        {
            Parallel.For(0, array.Length - 1, new ParallelOptions { MaxDegreeOfParallelism = maxThreads }, idx =>
                {
                    array[idx] = 0;
                });
        }
    }
}
