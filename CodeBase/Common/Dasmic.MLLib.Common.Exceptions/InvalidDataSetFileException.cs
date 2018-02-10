using System;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class InvalidDataSetFileException:Exception
    {
        public InvalidDataSetFileException( string message,
                    Exception innerException) : base(message, innerException)
        {

        }

        public InvalidDataSetFileException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_invalid_data_set_file;
            }
        }
    }
}
