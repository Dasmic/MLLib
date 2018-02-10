using System;
using System.IO;

namespace Dasmic.MLLib.Common.IO.Windows
{
    public class TextFileWriter:IDisposable
    {
        StreamWriter _writer;
         
        public TextFileWriter(string filePath)
        {
            try
            {
                _writer = new StreamWriter(
                    File.Open(filePath,FileMode.Create));
                _writer.AutoFlush=true; //Will flush after every write
            }
            catch(Exception ex)
            {
                _writer = null;
                throw ex;
            }
            finally
            {
                
            }
        }

        public void Dispose()
        {
            CompleteWrite();
        }

        /// <summary>
        /// Writes a line  with a carriage return
        /// </summary>
        /// <param name="line"></param>
        public void WriteLine(string line)
        {
            _writer.Write(line + "\r\n");
        }

        public void CompleteWrite()
        {
             //_writer.Close();
            _writer.Dispose();
        }


    }
}
