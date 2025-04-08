const std = @import("std");

const debug = false;
const allowed = .{
  .ok = true, // hack to make this always a struct
  .@"cnn.zig" = true,
  // .@"layers.zig" = true,
  // .@"functions.zig" = true,
  // .@"read_minst.zig" = true,
};

fn isAllowed(comptime src: *const std.builtin.SourceLocation) bool {
  inline for (@typeInfo(@TypeOf(allowed)).@"struct".fields) |field| {
    if (comptime !std.mem.eql(u8, field.name, src.file)) continue;
    const val = @field(allowed, field.name);
    if (@TypeOf(val) == bool) return val;

    inline for (@typeInfo(@TypeOf(val)).@"struct".fields) |field2| {
      if (comptime !std.mem.eql(u8, field2.name, src.fn_name)) continue;
      return @field(val, field2.name);
    }
  }
  return false;
}

var gpa: if (debug) std.heap.GeneralPurposeAllocator(.{}) else void = .{};
var donemap: if (debug) std.AutoHashMap(usize, void) else void = undefined;
pub fn init() void {
  if (debug) {
    donemap = @TypeOf(donemap).init(gpa.allocator());
  }
}

const stdout = std.io.getStdOut().writer();
pub var buffered = std.io.bufferedWriter(stdout);
pub const writer = buffered.writer();

fn printSrc(src: *const std.builtin.SourceLocation) void {
  writer.print("{s}:{d}:{d} -> {s}", .{ src.file, src.line, src.column, src.fn_name }) catch {};
}

pub fn log(comptime src: *const std.builtin.SourceLocation, comptime format: []const u8, args: anytype) void {
  if (debug) {
    if (!isAllowed(src)) {
      const gp = donemap.getOrPut(@intFromPtr(src)) catch return;
      if (gp.found_existing) return;
      writer.print("!! ", .{}) catch {};
      printSrc(src);
      writer.print("\n", .{}) catch {};
    } else {
      writer.print(">> ", .{}) catch {};
      printSrc(src);
      writer.print("\n", .{}) catch {};
      writer.print(format, args) catch {};
    }
  } else {
    if (!isAllowed(src)) return;
    writer.print(format, args) catch {};
  }
  buffered.flush() catch {};
}


pub fn lognoflush(comptime src: *const std.builtin.SourceLocation, comptime format: []const u8, args: anytype) void {
  if (!isAllowed(src)) return;
  writer.print(format, args) catch {};
  flushlog(src);
}

pub fn flushlog(comptime src: *const std.builtin.SourceLocation) void {
  if (!isAllowed(src)) return;
  buffered.flush() catch {};
}

