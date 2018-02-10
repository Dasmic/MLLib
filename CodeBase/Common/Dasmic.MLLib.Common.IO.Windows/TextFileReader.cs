using System;
using System.IO;

namespace Dasmic.MLLib.Common.IO.Windows
{
    public class TextFileReader
    {
        StreamReader _reader;

        public TextFileReader(string filePath)
        {
            try
            {
                _reader = new StreamReader(
                    File.Open(filePath, FileMode.Open));                
            }
            catch (Exception ex)
            {
                _reader = null;
                throw ex;
            }
            finally
            {

            }
        }

        public void Dispose()
        {
            CompleteRead();
        }

        /// <summary>
        /// Reads and retruns a single line
        /// </summary>
        /// <param name="line"></param>
        public string ReadLine()
        {
            string singleLine = _reader.ReadLine();
            return singleLine;
        }

        public void CompleteRead()
        {
            if(_reader != null)
                _reader.Close();
            _reader.Dispose();
        }
    }
}
