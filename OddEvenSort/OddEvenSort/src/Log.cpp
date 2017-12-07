//--------------------------------------------------------------------------------------
// File: Log.h
// Project: Function library
//
// Author Mattias Fredriksson 2017.
//--------------------------------------------------------------------------------------

#include"Log.h"
#include<fstream>
#include<chrono>
#include<iomanip>
#include<time.h>

namespace mf {

	Log::Log() :
		file(), logDate(), initialized(false) {
	}
	Log::Log(const std::string &logFile, bool clearLogFile, bool logDate, bool initMsg) :
		initialized(false) {
		initLog(logFile, clearLogFile, logDate, initMsg);
	}

	void Log::initLog(const std::string &logFile, bool clearLogFile, bool logDate, bool initMsg) {
		this->file = logFile;
		this->logDate = logDate;
		this->initialized = true;
		if (clearLogFile)
			clearLog();
		if (initMsg)
			logMsg("Log initialized " + (logDate ? "" : getDate()));
	}

	Log::~Log() {}

	void Log::logMsg(const std::string &msg) {
		isInit();
		std::ofstream log_file(file, std::ios_base::out | std::ios_base::app);

		//Print log msg
		if (logDate)
			log_file << getDate() << ' ';
		log_file << msg << std::endl;
	}


	void Log::clearLog() {
		isInit();
		std::ofstream log_file(file, std::ios_base::out | std::ios_base::trunc);
	}

	std::string Log::getDate() {

		const int parseSize = 20;
		char tArr[parseSize + 2];
		//Get the date & time
		auto now = std::chrono::system_clock::now();
		auto in_time_t = std::chrono::system_clock::to_time_t(now);
		tm t;
		//Parse it to string
		localtime_s(&t, &in_time_t);
		std::strftime(&tArr[1], 20, "%Y-%m-%d %H:%M:%S", &t);

		//Add some clamps to the string
		tArr[0] = '<'; tArr[parseSize] = '>'; tArr[parseSize + 1] = '\0';

		return std::string(tArr);
	}

	void Log::setDate(bool logDate) {
		this->logDate = logDate;
	}
}
