using System.Linq;
using Dasmic.Portable.Core;

namespace Dasmic.MLLib.Common.DataManagement
{
    public static class ArrayManipulation
    {
        #region Size Reduction
        /// <summary>
        /// Returns a new 2D array with specific column removed
        /// </summary>
        /// <param name="origArray"></param>
        /// <param name="columnIdx"></param>
        /// <returns></returns>
        public static T[][] RemoveSpecificColumn2D<T>(T [][] origArray, int columnIdx)
        {
            if (columnIdx < 0 || columnIdx > origArray.Length - 1)
                return origArray;

            T[][] newArray = new T[origArray.Length-1][];
            int newIdx = 0;
            for (int col=0;col< origArray.Length;col++)
            {
                if(col != columnIdx)
                {
                    newArray[newIdx++] = origArray[col];
                }
            }
            return newArray;
        }

        /// <summary>
        /// Returns a new 2D array after removing  last column in a 2D
        /// </summary>
        /// <param name="origArray"></param>
        /// <returns></returns>
        public static T[][] RemoveLastColumn2D<T>(T[][] origArray)
        {
            return RemoveSpecificColumn2D(origArray, origArray.Length - 1);
        }

        /// <summary>
        /// Removes specified index value from a 1D array and retrurns 
        /// new array with value removed
        /// </summary>
        /// <param name="origArray">Array which needs to be modified </param>
        /// <param name="idx">Index to remove</param>
        /// <returns></returns>
        public static T[] RemoveSpecificColumn1D<T>(T[] origArray, int idx)
        {
            if (idx < 0 || idx > origArray.Length - 1)
                return origArray;

            T[] newArray = new T[origArray.Length - 1];
            int newIdx = 0;
            for (int col = 0; col < origArray.Length; col++)
            {
                if (col != idx)
                {
                    newArray[newIdx++] = origArray[col];
                }
            }
            return newArray;
        }

        /// <summary>
        /// Removes the last column from a 1D array and returns modified array 
        /// </summary>
        /// <param name="origArray"></param>
        /// <returns></returns>
        public static T[] RemoveLastColumn1D<T>(T[] origArray)
        {
            return RemoveSpecificColumn1D(origArray, origArray.Length - 1);
        }

        #endregion Size Reduction


        /// <summary>
        /// Returns specific column in 2D array as 1D array
        /// </summary>
        /// <param name="origArray"></param>
        /// <param name="colIdx">0 based column index</param>
        /// <returns></returns>
        public static T[] GetColAs1DArray<T>(T[][] origArray,
                                   long colIdx)
        {
            return origArray[colIdx];
        }

        /// <summary>
        /// Returns last column in 2D array as 1D array
        /// </summary>
        /// <param name="origArray"></param>        
        /// <returns></returns>
        public static T[] GetLastColAs1DArray<T>(T[][] origArray)
        {
            return GetColAs1DArray(origArray,origArray.Length-1);
        }

        /// <summary>
        /// Returns a single row in 2D array as a 1D array uptil specified column index
        /// </summary>
        /// <param name="origArray">Original array</param>
        /// <param name="rowIdx">0 based row idx that needs to be converted</param>        
        /// <param name="endColumnIdx">Column Idx till which data has to be copied</param>        
        /// <returns></returns>
        public static T[] GetRowAs1DArray<T>(T[][] origArray,
                                   long rowIdx,  int endColumnIdx)
        {            
                T[] data = new T[endColumnIdx + 1];
                for (int col = 0; col <= endColumnIdx; col++)
                {
                    data[col] =origArray[col][rowIdx];
                }
                return data;            
        }


        /// <summary>
        /// Returns a single row in 2D array as a 1D array
        /// </summary>
        /// <typeparam name="T">Can take any Generic type</typeparam>
        /// <param name="origArray"></param>
        /// <param name="rowIdx"></param>
        /// <returns></returns>
        public static T[] GetRowAs1DArray<T>(T[][] origArray,
                                   long rowIdx)
        {
            return GetRowAs1DArray(origArray, rowIdx, origArray.Length - 1);
        }

    }
}
