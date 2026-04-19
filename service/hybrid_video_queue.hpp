#pragma once

#include <fstream>
#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>

class HybridVideoQueue {
private:
    std::queue<cv::Mat> mem_queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    std::atomic<bool> stop_flag_{ false };

    size_t mem_limit_;
    std::string temp_file_path_;

    std::fstream file_stream_;
    size_t disk_write_idx_ = 0;
    size_t disk_read_idx_ = 0;

    int width_, height_, type_;
    size_t frame_size_;

public:
    HybridVideoQueue(size_t mem_limit, const std::string& temp_file, int w, int h, int type)
        : mem_limit_(mem_limit), temp_file_path_(temp_file),
        width_(w), height_(h), type_(type) {
        frame_size_ = static_cast<size_t>(w) * h * CV_ELEM_SIZE(type);
        file_stream_.open(temp_file_path_, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
        if (!file_stream_.is_open()) {
            std::cerr << "[HybridVideoQueue] Failed to create temp file: " << temp_file_path_ << std::endl;
        }
    }

    ~HybridVideoQueue() {
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
        std::remove(temp_file_path_.c_str());
    }

    void push(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!frame.isContinuous()) {
            std::cerr << "[HybridVideoQueue] Error: Frame must be continuous!" << std::endl;
            return;
        }

        if (mem_queue_.size() < mem_limit_ && disk_write_idx_ == disk_read_idx_) {
            mem_queue_.push(frame.clone());
            // std::cout << "[algoqueue]write into memory|frame_queue size: "<< mem_queue_.size() << std::endl;
        }
        else {
            // Write to disk file
            file_stream_.clear(); // Clear EOF flags if any
            file_stream_.seekp(disk_write_idx_ * frame_size_, std::ios::beg);
            file_stream_.write(reinterpret_cast<const char*>(frame.data), frame_size_);
            file_stream_.flush(); // Write to disk directly
            disk_write_idx_++;
            // std::cout << "[FrameQueue]write into binary file" << std::endl;
        }
        cond_var_.notify_one();
    }

    bool wait_and_pop(cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this] {
            return !mem_queue_.empty() || disk_read_idx_ < disk_write_idx_ || stop_flag_;
            });

        if (mem_queue_.empty() && disk_read_idx_ == disk_write_idx_ && stop_flag_) {
            return false;
        }

        if (!mem_queue_.empty()) {
            frame = mem_queue_.front();
            mem_queue_.pop();
        }
        else if (disk_read_idx_ < disk_write_idx_) {
            frame.create(height_, width_, type_);
            file_stream_.clear();
            file_stream_.seekg(disk_read_idx_ * frame_size_, std::ios::beg);
            file_stream_.read(reinterpret_cast<char*>(frame.data), frame_size_);
			// std::cout << "[FrameQueue]consume from algo queue!|write_idx:" << disk_write_idx_ << "|read_idx:" << disk_read_idx_ << std::endl;
            disk_read_idx_++;

            // Reset disk file to save space when empty
            if (disk_read_idx_ == disk_write_idx_) {
                disk_read_idx_ = 0;
                disk_write_idx_ = 0;
                file_stream_.close();
                file_stream_.open(temp_file_path_, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
                std::cout << "Finish disk processing!" << std::endl;
            }
        }
        return true;
    }

    void stop() {
        stop_flag_ = true;
        cond_var_.notify_all();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return mem_queue_.size() + (disk_write_idx_ - disk_read_idx_);
    }
};
