#pragma once
//--------------------------------------------------------------------------------------
// File: Log.h
// Project: Function library
//
// Author Mattias Fredriksson 2017.
//--------------------------------------------------------------------------------------

#include<string>
#include<stdexcpt.h>

namespace mf {

	class Log {

	private:
		/*File name for the log file*/
		std::string file;
		/*If date should be logged with each message*/
		bool logDate;
		/*If log is intialized*/
		bool initialized;

		inline void isInit() {
			if (!initialized)
				throw  std::invalid_argument("Log not initialized");
		}

	public:

		/*Gets the current date and time as a string*/
		static std::string getDate();
		/*An uninitialized log file which when initialized can print log messages*/
		Log();
		/*
		Creates a log file which can print log messages.
		logFile			>> File name and/or directory for the log file
		clearLogFile	>> If logfile should be cleared
		logDate			>> Add date/time to each log message
		initMsg			>> Initialization message
		*/
		Log(const std::string &logFile, bool clearLogFile = false, bool logDate = true, bool initMsg = true);

		/*
		Initialize a log file object which can print log messages.
		logFile			>> File name and/or directory for the log file
		clearLogFile	>> If logfile should be cleared
		logDate			>> Add date/time to each log message
		initMsg			>> Initialization message
		*/
		void initLog(const std::string &logFile, bool clearLogFile = false, bool logDate = true, bool initMsg = true);
		virtual ~Log();

		/*Log a message*/
		void logMsg(const std::string &msg);
		/*Clear the entire log*/
		void clearLog();
		/*Enables or Disable log date*/
		void setDate(bool logDate);
	};

}