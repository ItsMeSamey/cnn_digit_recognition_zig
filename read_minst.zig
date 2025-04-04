const std = @import("std");

/// This function generates a struct that can be used to load minst dataset 
pub fn GetMinstIterator(comptime ROWS: u32, comptime COLS: u32) type {
  return struct {
    // The data in this dataset
    data: [*]u8,
    // Labels in this dataset
    labels: [*]u8,
    // how many images are present in this dataset
    count: u32,
    // What index are we at
    index: u32,

    pub const InitError = error {
      InvalidImagesFile, InvalidLabelsFile,
      InvalidImagesFileMagic, IncompatibleImagesFile,
      InvalidLabelsFileMagic, IncompatibleLabelsFile,
      IncompatibleFiles,
    } || std.fs.File.OpenError || std.fs.File.StatError || std.fs.File.ReadError || std.mem.Allocator.Error || error{
      EndOfStream,
    };
      
    pub fn init(images: []const u8, labels: []const u8, allocator: std.mem.Allocator) InitError!@This() {
      var images_file = try std.fs.cwd().openFile(images, .{});
      defer images_file.close();
      var labels_file = try std.fs.cwd().openFile(labels, .{});
      defer labels_file.close();

      const images_stats = try images_file.stat();
      if (images_stats.size < 16) return InitError.InvalidImagesFile;
      const labels_stats = try labels_file.stat();
      if (labels_stats.size < 8) return InitError.InvalidLabelsFile;

      const images_header = try images_file.reader().readStructEndian(packed struct {
        magic: u32,
        num_images: u32,
        rows: u32,
        cols: u32,
      }, std.builtin.Endian.big);
      if (images_header.magic != 2051) return InitError.InvalidImagesFileMagic;
      if (images_header.rows != ROWS or images_header.cols != COLS) return InitError.IncompatibleImagesFile;

      const labels_header = try labels_file.reader().readStructEndian(packed struct {
        magic: u32,
        num_labels: u32,
      }, std.builtin.Endian.big);
      if (labels_header.magic != 2049) return InitError.InvalidLabelsFileMagic;
      if (labels_stats.size != labels_header.num_labels + 8) return InitError.IncompatibleLabelsFile;

      if (images_header.num_images != labels_header.num_labels) return InitError.IncompatibleFiles;

      const allocation = try allocator.alloc(u8, images_stats.size - 16 + labels_stats.size - 8);
      errdefer allocator.free(allocation);

      const image_file_read_count = try images_file.readAll(allocation[0..images_stats.size - 16]);
      std.debug.assert(image_file_read_count == images_stats.size - 16);
      const labels_file_read_count = try labels_file.readAll(allocation[images_stats.size - 16..]);
      std.debug.assert(labels_file_read_count == labels_stats.size - 8);

      return .{
        .data = allocation[0..images_stats.size - 16].ptr,
        .labels = allocation[images_stats.size - 16..].ptr,
        .count = images_header.num_images,
        .index = 0,
      };
    }

    pub fn free(self: *const @This(), allocator: std.mem.Allocator) void {
      allocator.free(self.data[0..self.count * ROWS * COLS + self.count]);
    }

    pub fn next(self: *@This()) ?struct{
      image: * align(1) [ROWS][COLS]u8,
      label: u8,
    } {
      if (self.index >= self.count) return null;
      defer self.index += 1;
      return .{
        .image = @ptrCast(self.data[self.index * (ROWS * COLS) ..][0..ROWS * COLS].ptr),
        .label = self.labels[self.index],
      };
    }

    pub fn hasNext(self: *@This()) bool {
      return self.index >= self.count;
    }

    pub fn reset(self: *@This()) void {
      self.index = 0;
    }
  };
}

const stdout = std.io.getStdOut().writer();
pub fn printImage(img: anytype) void {
  const ascii_chars = [_]u8{
    ' ', '.', ',', '\'', ':', '`', 't', '-', '_', '!', 'i', 'l', '|', ';',
    'f', 'j', 'r', 'c', 's', '(', ')', '{', '}', '[', ']', '<', '>', '/', '\\', '1',
    '+', '=', '~', '*', 'n', 'u', 'v', 'x', 'y', 'z', '2', '3', '4', '5', '6', '7',
    'a', 'e', 'o', 'w', 'k', 'h', 'p', 'q', 'd', 'b', '9', '0',
    'A', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'P', 'R', 'Q', 'D', 'B',
    'H', 'K', 'L',
    '#',
    '@', '%', '&', '$'
  };
  const num_chars = ascii_chars.len;
  if (img.len == 0 or img[0].len == 0) return;

  comptime var T = @TypeOf(img);
  if (comptime @typeInfo(T) == .pointer) T = std.meta.Child(T);
  T = std.meta.Child(std.meta.Child(T));
  var min_val: T = img[0][0];
  var max_val: T = img[0][0];

  for (img) |row| {
    for (row) |item| {
      if (item < min_val) {
        min_val = item;
      } else if (item > max_val) {
        max_val = item;
      }
    }
  }

  const writer = stdout;

  const diff = max_val - min_val;
  const factor = @as(f32, @floatFromInt(ascii_chars.len - 1)) / @as(f32, if (@typeInfo(T) == .float) @floatFromInt(diff) else @floatFromInt(diff));
  if (min_val == max_val) {
    for (img) |row| {
      for (row) |_| {
        writer.print("{c}", .{ascii_chars[num_chars/2]}) catch {};
      }
      writer.print("\n", .{}) catch {};
    }
    return;
  }

  for (img) |row| {
    for (row) |item| {
      const index_float = factor * @as(f32, if (@typeInfo(T) == .float) @floatCast(item - min_val) else @floatFromInt(item - min_val));
      writer.print("{c}", .{ascii_chars[@intFromFloat(@round(index_float))]}) catch {};
    }
    writer.print("\n", .{}) catch {};
  }
}

test "GetMinstIterator" {
  const allocator = std.testing.allocator;
  const MNIST = GetMinstIterator(28, 28);
  var mnist_train = try MNIST.init("./datasets/train-images.idx3-ubyte", "./datasets/train-labels.idx1-ubyte", allocator);
  defer mnist_train.free(allocator);
  printImage(mnist_train.next().?.image);

  var mnist_test = try MNIST.init("./datasets/t10k-images.idx3-ubyte", "./datasets/t10k-labels.idx1-ubyte", allocator);
  defer mnist_test.free(allocator);
  printImage(mnist_test.next().?.image);
}

