#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

struct HsvRange {
	cv::Scalar lo;
	cv::Scalar hi;
};

struct Circle {
	int x;
	int y;
	int r;
};

struct DiceState {
	int yellowDotsOnRed = -1;
	int redDotsOnYellow = -1;
	std::string whiteDieColor = "unknown";

	bool operator==(const DiceState &other) const {
		return yellowDotsOnRed == other.yellowDotsOnRed &&
			   redDotsOnYellow == other.redDotsOnYellow &&
			   whiteDieColor == other.whiteDieColor;
	}

	bool complete() const {
		return yellowDotsOnRed >= 0 && redDotsOnYellow >= 0 && whiteDieColor != "unknown";
	}
};

struct TrayTuning {
	int hLow = 3;
	int hHigh = 35;
	int sLow = 25;
	int sHigh = 255;
	int vLow = 20;
	int vHigh = 255;

	int minAreaPct = 3;
	int minCircularityPct = 28;
	int minRadiusPct = 12;
	int maskShrinkPct = 95;
};

struct TrayDebugInfo {
	int contourCount = 0;
	double bestAreaFrac = 0.0;
	double bestCircularity = 0.0;
	double bestRadiusFrac = 0.0;
};

static const std::vector<HsvRange> kRedDie = {
	{{0, 120, 70}, {10, 255, 255}},
	{{160, 120, 70}, {180, 255, 255}},
};

static const std::vector<HsvRange> kYellowDie = {
	{{15, 100, 100}, {35, 255, 255}},
};

static const std::vector<HsvRange> kWhiteDie = {
	{{0, 0, 170}, {180, 50, 255}},
};

static const std::vector<HsvRange> kYellowPips = {
	{{15, 100, 100}, {35, 255, 255}},
};

static const std::vector<HsvRange> kRedPips = {
	{{0, 100, 70}, {10, 255, 255}},
	{{160, 100, 70}, {180, 255, 255}},
};

static const std::vector<HsvRange> kWhiteSymbolBlack = {
	{{0, 0, 0}, {180, 255, 70}},
};

static const std::vector<HsvRange> kWhiteSymbolYellow = {
	{{15, 80, 80}, {35, 255, 255}},
};

static const std::vector<HsvRange> kWhiteSymbolGreen = {
	{{40, 60, 60}, {80, 255, 255}},
};

static const std::vector<HsvRange> kWhiteSymbolBlue = {
	{{90, 80, 60}, {130, 255, 255}},
};

cv::Mat maskFromRanges(const cv::Mat &hsv, const std::vector<HsvRange> &ranges) {
	cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8U);
	for (const auto &r : ranges) {
		cv::Mat part;
		cv::inRange(hsv, r.lo, r.hi, part);
		cv::bitwise_or(mask, part, mask);
	}
	return mask;
}

cv::Mat maskFromSingleRange(const cv::Mat &hsv, int hLow, int hHigh, int sLow, int sHigh, int vLow, int vHigh) {
	const cv::Scalar lo(hLow, sLow, vLow);
	const cv::Scalar hi(hHigh, sHigh, vHigh);
	cv::Mat mask;
	cv::inRange(hsv, lo, hi, mask);
	return mask;
}

void createTrayControls(const std::string &windowName, TrayTuning &tuning) {
	cv::createTrackbar("Tray H low", windowName, nullptr, 180);
	cv::createTrackbar("Tray H high", windowName, nullptr, 180);
	cv::createTrackbar("Tray S low", windowName, nullptr, 255);
	cv::createTrackbar("Tray S high", windowName, nullptr, 255);
	cv::createTrackbar("Tray V low", windowName, nullptr, 255);
	cv::createTrackbar("Tray V high", windowName, nullptr, 255);
	cv::createTrackbar("Tray min area %", windowName, nullptr, 40);
	cv::createTrackbar("Tray circularity %", windowName, nullptr, 100);
	cv::createTrackbar("Tray min radius %", windowName, nullptr, 40);
	cv::createTrackbar("Tray mask shrink %", windowName, nullptr, 100);

	cv::setTrackbarPos("Tray H low", windowName, tuning.hLow);
	cv::setTrackbarPos("Tray H high", windowName, tuning.hHigh);
	cv::setTrackbarPos("Tray S low", windowName, tuning.sLow);
	cv::setTrackbarPos("Tray S high", windowName, tuning.sHigh);
	cv::setTrackbarPos("Tray V low", windowName, tuning.vLow);
	cv::setTrackbarPos("Tray V high", windowName, tuning.vHigh);
	cv::setTrackbarPos("Tray min area %", windowName, tuning.minAreaPct);
	cv::setTrackbarPos("Tray circularity %", windowName, tuning.minCircularityPct);
	cv::setTrackbarPos("Tray min radius %", windowName, tuning.minRadiusPct);
	cv::setTrackbarPos("Tray mask shrink %", windowName, tuning.maskShrinkPct);
}

void readTrayControls(const std::string &windowName, TrayTuning &tuning) {
	tuning.hLow = cv::getTrackbarPos("Tray H low", windowName);
	tuning.hHigh = cv::getTrackbarPos("Tray H high", windowName);
	tuning.sLow = cv::getTrackbarPos("Tray S low", windowName);
	tuning.sHigh = cv::getTrackbarPos("Tray S high", windowName);
	tuning.vLow = cv::getTrackbarPos("Tray V low", windowName);
	tuning.vHigh = cv::getTrackbarPos("Tray V high", windowName);
	tuning.minAreaPct = cv::getTrackbarPos("Tray min area %", windowName);
	tuning.minCircularityPct = cv::getTrackbarPos("Tray circularity %", windowName);
	tuning.minRadiusPct = cv::getTrackbarPos("Tray min radius %", windowName);
	tuning.maskShrinkPct = cv::getTrackbarPos("Tray mask shrink %", windowName);
}

void clampTrayTuning(TrayTuning &tuning) {
	tuning.hLow = std::clamp(tuning.hLow, 0, 180);
	tuning.hHigh = std::clamp(tuning.hHigh, 0, 180);
	if (tuning.hLow > tuning.hHigh) {
		std::swap(tuning.hLow, tuning.hHigh);
	}

	tuning.sLow = std::clamp(tuning.sLow, 0, 255);
	tuning.sHigh = std::clamp(tuning.sHigh, 0, 255);
	if (tuning.sLow > tuning.sHigh) {
		std::swap(tuning.sLow, tuning.sHigh);
	}

	tuning.vLow = std::clamp(tuning.vLow, 0, 255);
	tuning.vHigh = std::clamp(tuning.vHigh, 0, 255);
	if (tuning.vLow > tuning.vHigh) {
		std::swap(tuning.vLow, tuning.vHigh);
	}

	tuning.minAreaPct = std::clamp(tuning.minAreaPct, 1, 40);
	tuning.minCircularityPct = std::clamp(tuning.minCircularityPct, 10, 100);
	tuning.minRadiusPct = std::clamp(tuning.minRadiusPct, 3, 40);
	tuning.maskShrinkPct = std::clamp(tuning.maskShrinkPct, 70, 100);
}

std::optional<Circle> findTrayCircle(const cv::Mat &frameBgr, const TrayTuning &tuning, cv::Mat *debugMask = nullptr,
									 TrayDebugInfo *debugInfo = nullptr) {
	cv::Mat hsv;
	cv::cvtColor(frameBgr, hsv, cv::COLOR_BGR2HSV);

	cv::Mat mask = maskFromSingleRange(hsv,
								  tuning.hLow, tuning.hHigh,
								  tuning.sLow, tuning.sHigh,
								  tuning.vLow, tuning.vHigh);
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
					 cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)),
					 cv::Point(-1, -1), 2);
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
					 cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
					 cv::Point(-1, -1), 1);

	if (debugMask != nullptr) {
		*debugMask = mask.clone();
	}

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	if (debugInfo != nullptr) {
		debugInfo->contourCount = static_cast<int>(contours.size());
	}

	const double frameArea = static_cast<double>(frameBgr.rows * frameBgr.cols);
	const double minAreaFrac = static_cast<double>(tuning.minAreaPct) / 100.0;
	const double minCircularity = static_cast<double>(tuning.minCircularityPct) / 100.0;
	const double minRadiusFrac = static_cast<double>(tuning.minRadiusPct) / 100.0;
	double bestScore = -1.0;
	std::optional<Circle> best;
	double bestAreaFracSeen = 0.0;
	double bestCircularitySeen = 0.0;
	double bestRadiusFracSeen = 0.0;

	for (const auto &c : contours) {
		const double a = cv::contourArea(c);
		if (a < frameArea * minAreaFrac) {
			continue;
		}
		const double p = cv::arcLength(c, true);
		if (p <= 0.0) {
			continue;
		}
		const double circularity = 4.0 * CV_PI * a / (p * p);
		if (circularity < minCircularity) {
			continue;
		}

		cv::Point2f center;
		float radius = 0.0F;
		cv::minEnclosingCircle(c, center, radius);
		const double radiusFrac = static_cast<double>(radius) / static_cast<double>(std::min(frameBgr.rows, frameBgr.cols));
		if (radius < static_cast<float>(std::min(frameBgr.rows, frameBgr.cols)) * static_cast<float>(minRadiusFrac)) {
			continue;
		}

		const double score = a * circularity;
		if (score > bestScore) {
			bestScore = score;
			bestAreaFracSeen = a / frameArea;
			bestCircularitySeen = circularity;
			bestRadiusFracSeen = radiusFrac;
			best = Circle{static_cast<int>(center.x), static_cast<int>(center.y), static_cast<int>(radius)};
		}
	}

	if (debugInfo != nullptr) {
		debugInfo->bestAreaFrac = bestAreaFracSeen;
		debugInfo->bestCircularity = bestCircularitySeen;
		debugInfo->bestRadiusFrac = bestRadiusFracSeen;
	}

	return best;
}

cv::Mat trayMaskFromCircle(const cv::Size &sz, const Circle &c, double shrink = 0.95) {
	cv::Mat mask = cv::Mat::zeros(sz, CV_8U);
	const int rr = std::max(1, static_cast<int>(c.r * shrink));
	cv::circle(mask, cv::Point(c.x, c.y), rr, cv::Scalar(255), -1);
	return mask;
}

std::optional<std::vector<cv::Point>> detectDieContour(const cv::Mat &hsv,
													   const std::vector<HsvRange> &dieBodyRange,
													   const cv::Mat &trayMask) {
	cv::Mat mask = maskFromRanges(hsv, dieBodyRange);
	cv::bitwise_and(mask, trayMask, mask);

	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
					 cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)),
					 cv::Point(-1, -1), 2);
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
					 cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)),
					 cv::Point(-1, -1), 1);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	const double frameArea = static_cast<double>(hsv.rows * hsv.cols);
	const double minArea = frameArea * 0.003;
	const double maxArea = frameArea * 0.04;

	double bestArea = 0.0;
	std::optional<std::vector<cv::Point>> best;

	for (const auto &c : contours) {
		const double a = cv::contourArea(c);
		if (a < minArea || a > maxArea) {
			continue;
		}

		cv::RotatedRect rr = cv::minAreaRect(c);
		if (rr.size.width <= 0.0F || rr.size.height <= 0.0F) {
			continue;
		}

		const double ar = std::max(rr.size.width, rr.size.height) / std::min(rr.size.width, rr.size.height);
		if (ar > 1.55) {
			continue;
		}

		const double rectArea = rr.size.width * rr.size.height;
		const double fill = a / std::max(rectArea, 1.0);
		if (fill < 0.58) {
			continue;
		}

		std::vector<cv::Point2f> rectPts(4);
		rr.points(rectPts.data());
		std::vector<cv::Point> rectPtsInt;
		rectPtsInt.reserve(4);
		for (const auto &p : rectPts) {
			rectPtsInt.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
		}
		const double shape = cv::matchShapes(c, rectPtsInt, cv::CONTOURS_MATCH_I1, 0.0);
		if (shape > 0.16) {
			continue;
		}

		if (a > bestArea) {
			bestArea = a;
			best = c;
		}
	}

	return best;
}

int countCircularBlobs(const cv::Mat &binaryMask, double minAreaFrac, double maxAreaFrac) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	const double areaAll = static_cast<double>(binaryMask.rows * binaryMask.cols);
	const double minArea = areaAll * minAreaFrac;
	const double maxArea = areaAll * maxAreaFrac;

	int count = 0;
	for (const auto &c : contours) {
		const double a = cv::contourArea(c);
		if (a < minArea || a > maxArea) {
			continue;
		}
		const double p = cv::arcLength(c, true);
		if (p <= 0.0) {
			continue;
		}
		const double circ = 4.0 * CV_PI * a / (p * p);
		if (circ >= 0.35) {
			count++;
		}
	}
	return count;
}

int countPipsInDie(const cv::Mat &frameBgr,
				  const std::vector<cv::Point> &dieContour,
				  const std::vector<HsvRange> &pipRanges) {
	cv::Mat dieMask = cv::Mat::zeros(frameBgr.size(), CV_8U);
	cv::drawContours(dieMask, std::vector<std::vector<cv::Point>>{dieContour}, -1, cv::Scalar(255), -1);

	cv::Rect bbox = cv::boundingRect(dieContour);
	const int dx = static_cast<int>(bbox.width * 0.15);
	const int dy = static_cast<int>(bbox.height * 0.15);
	bbox.x += dx;
	bbox.y += dy;
	bbox.width = std::max(1, bbox.width - 2 * dx);
	bbox.height = std::max(1, bbox.height - 2 * dy);
	bbox &= cv::Rect(0, 0, frameBgr.cols, frameBgr.rows);

	cv::Mat hsv;
	cv::cvtColor(frameBgr, hsv, cv::COLOR_BGR2HSV);
	cv::Mat pipMask = maskFromRanges(hsv, pipRanges);
	cv::bitwise_and(pipMask, dieMask, pipMask);

	cv::Mat roiMask = cv::Mat::zeros(frameBgr.size(), CV_8U);
	cv::rectangle(roiMask, bbox, cv::Scalar(255), -1);
	cv::bitwise_and(pipMask, roiMask, pipMask);

	cv::morphologyEx(pipMask, pipMask, cv::MORPH_OPEN,
					 cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

	const cv::Mat pipRoi = pipMask(bbox);
	return countCircularBlobs(pipRoi, 0.003, 0.12);
}

std::string detectWhiteDieColor(const cv::Mat &frameBgr, const std::vector<cv::Point> &whiteContour) {
	cv::Rect bbox = cv::boundingRect(whiteContour);
	const int dx = static_cast<int>(bbox.width * 0.2);
	const int dy = static_cast<int>(bbox.height * 0.2);
	bbox.x += dx;
	bbox.y += dy;
	bbox.width = std::max(1, bbox.width - 2 * dx);
	bbox.height = std::max(1, bbox.height - 2 * dy);
	bbox &= cv::Rect(0, 0, frameBgr.cols, frameBgr.rows);

	cv::Mat hsv;
	cv::cvtColor(frameBgr, hsv, cv::COLOR_BGR2HSV);
	cv::Mat roi = hsv(bbox);

	struct ColorScore {
		std::string name;
		int score;
	};

	std::vector<ColorScore> scores;
	scores.push_back({"black", cv::countNonZero(maskFromRanges(roi, kWhiteSymbolBlack))});
	scores.push_back({"yellow", cv::countNonZero(maskFromRanges(roi, kWhiteSymbolYellow))});
	scores.push_back({"green", cv::countNonZero(maskFromRanges(roi, kWhiteSymbolGreen))});
	scores.push_back({"blue", cv::countNonZero(maskFromRanges(roi, kWhiteSymbolBlue))});

	const int minPixels = static_cast<int>(bbox.area() * 0.025);
	std::string bestColor = "unknown";
	int bestScore = 0;

	for (const auto &s : scores) {
		if (s.score >= minPixels && s.score > bestScore) {
			bestScore = s.score;
			bestColor = s.name;
		}
	}

	return bestColor;
}

cv::VideoCapture openExternalCamera() {
	std::vector<int> preferred = {1, 2, 0};
	for (int idx : preferred) {
		cv::VideoCapture cap(idx, cv::CAP_DSHOW);
		if (!cap.isOpened()) {
			cap.open(idx);
		}
		if (cap.isOpened()) {
			cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
			cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
			cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
			std::cout << "Using camera index " << idx << std::endl;
			return cap;
		}
	}
	return cv::VideoCapture();
}

int main() {
	cv::VideoCapture cap = openExternalCamera();
	if (!cap.isOpened()) {
		std::cerr << "Could not open camera." << std::endl;
		return 1;
	}

	const std::string kWindowName = "Different Approach";
	cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);

	TrayTuning trayTuning;
	createTrayControls(kWindowName, trayTuning);

	std::optional<Circle> lastTray;
	int trayMiss = 0;
	const int trayKeepFrames = 45;

	DiceState lastObserved;
	bool hasObserved = false;
	bool printedForState = false;
	auto stateSince = std::chrono::steady_clock::now();

	std::cout << "Press q to quit." << std::endl;

	for (;;) {
		cv::Mat frame;
		if (!cap.read(frame) || frame.empty()) {
			std::cerr << "Could not read frame." << std::endl;
			break;
		}

		if (cv::getWindowProperty(kWindowName, cv::WND_PROP_VISIBLE) < 1) {
			break;
		}

		readTrayControls(kWindowName, trayTuning);
		clampTrayTuning(trayTuning);

		cv::Mat trayDebugMask;
		TrayDebugInfo trayDbg;
		const auto trayNow = findTrayCircle(frame, trayTuning, &trayDebugMask, &trayDbg);
		if (trayNow.has_value()) {
			lastTray = trayNow;
			trayMiss = 0;
		} else if (lastTray.has_value()) {
			trayMiss++;
			if (trayMiss > trayKeepFrames) {
				lastTray.reset();
			}
		}

		cv::Mat trayMask(frame.size(), CV_8U, cv::Scalar(0));
		if (lastTray.has_value()) {
			const double shrink = static_cast<double>(trayTuning.maskShrinkPct) / 100.0;
			trayMask = trayMaskFromCircle(frame.size(), *lastTray, shrink);
		}

		cv::Mat display = frame.clone();
		if (lastTray.has_value()) {
			cv::Mat outsideMask;
			cv::bitwise_not(trayMask, outsideMask);
			// Keep context visible but de-emphasize pixels outside the tray.
			display.setTo(cv::Scalar(35, 35, 35), outsideMask);
		}

		cv::Mat hsv;
		cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

		DiceState current;

		auto redDie = detectDieContour(hsv, kRedDie, trayMask);
		auto yellowDie = detectDieContour(hsv, kYellowDie, trayMask);
		auto whiteDie = detectDieContour(hsv, kWhiteDie, trayMask);

		if (redDie.has_value()) {
			current.yellowDotsOnRed = countPipsInDie(frame, *redDie, kYellowPips);
			cv::polylines(display, std::vector<std::vector<cv::Point>>{*redDie}, true, cv::Scalar(0, 0, 255), 2);
		}

		if (yellowDie.has_value()) {
			current.redDotsOnYellow = countPipsInDie(frame, *yellowDie, kRedPips);
			cv::polylines(display, std::vector<std::vector<cv::Point>>{*yellowDie}, true, cv::Scalar(0, 255, 255), 2);
		}

		if (whiteDie.has_value()) {
			current.whiteDieColor = detectWhiteDieColor(frame, *whiteDie);
			cv::polylines(display, std::vector<std::vector<cv::Point>>{*whiteDie}, true, cv::Scalar(255, 255, 255), 2);
		}

		if (lastTray.has_value()) {
			const double shrink = static_cast<double>(trayTuning.maskShrinkPct) / 100.0;
			cv::circle(display, cv::Point(lastTray->x, lastTray->y), static_cast<int>(lastTray->r * shrink),
					   cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
		}

		if (!trayDebugMask.empty()) {
			cv::Mat trayDebugBgr;
			cv::cvtColor(trayDebugMask, trayDebugBgr, cv::COLOR_GRAY2BGR);
			cv::resize(trayDebugBgr, trayDebugBgr, cv::Size(220, 140));
			trayDebugBgr.copyTo(display(cv::Rect(10, display.rows - trayDebugBgr.rows - 10,
											 trayDebugBgr.cols, trayDebugBgr.rows)));
			cv::putText(display, "Tray mask", cv::Point(14, display.rows - trayDebugBgr.rows - 16),
						cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
		}

		const auto now = std::chrono::steady_clock::now();
		if (!hasObserved || !(current == lastObserved)) {
			lastObserved = current;
			hasObserved = true;
			stateSince = now;
			printedForState = false;
		} else {
			const auto stableMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - stateSince).count();
			if (!printedForState && stableMs >= 500 && current.complete()) {
				std::cout << "Stable state (>=500ms): "
						  << "yellow dots on red die = " << current.yellowDotsOnRed << ", "
						  << "red dots on yellow die = " << current.redDotsOnYellow << ", "
						  << "white die color = " << current.whiteDieColor
						  << std::endl;
				printedForState = true;
			}
		}

		std::string line1 = "Red die (yellow pips): " +
							(current.yellowDotsOnRed >= 0 ? std::to_string(current.yellowDotsOnRed) : "?");
		std::string line2 = "Yellow die (red pips): " +
							(current.redDotsOnYellow >= 0 ? std::to_string(current.redDotsOnYellow) : "?");
		std::string line3 = "White die color: " + current.whiteDieColor;
		std::string line4 = std::string("Tray: ") + (lastTray.has_value() ? "FOUND" : "not found") +
							", contours=" + std::to_string(trayDbg.contourCount);
		std::string line5 = "Tray best area%=" + std::to_string(static_cast<int>(trayDbg.bestAreaFrac * 100.0)) +
							", circ=" + std::to_string(static_cast<int>(trayDbg.bestCircularity * 100.0)) +
							", rad%=" + std::to_string(static_cast<int>(trayDbg.bestRadiusFrac * 100.0));

		cv::putText(display, line1, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
		cv::putText(display, line2, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
		cv::putText(display, line3, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
		cv::putText(display, line4, cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 255, 200), 2, cv::LINE_AA);
		cv::putText(display, line5, cv::Point(20, 148), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(180, 255, 255), 2, cv::LINE_AA);

		cv::imshow(kWindowName, display);
		const int key = cv::waitKey(1);
		if (key == 'q' || key == 'Q') {
			break;
		}
	}

	cap.release();
	cv::destroyAllWindows();
	return 0;
}

