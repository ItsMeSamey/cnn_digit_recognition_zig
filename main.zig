const std = @import("std");
const CNN = @import("cnn.zig").CNN;
const meta = @import("meta.zig");
const Layer = @import("layers.zig");
const logger = @import("logger.zig");
const MNIST = @import("read_minst.zig");
const Loss = @import("functions_loss.zig");
const Activation = @import("functions_activate.zig");

const cnn = CNN(f64, 28, 28, Loss.MeanSquaredError, [_]Layer.LayerType{
  Layer.mergeAny([_]Layer.LayerType{
    Layer.mergeAny(.{
      Layer.getConvolver(14,14, 7, 7, Activation.getPReLU(0.25)),
      Layer.getConvolver(14,14, 7, 7, Activation.getPReLU(0.25)),
      Layer.getConvolver(14,14, 7, 7, Activation.getPReLU(0.25)),
      Layer.getConvolver(14,14, 7, 7, Activation.getPReLU(0.25)),
    }),
  }),
  // Layer.getReshaper(14, 14),
  // Layer.mergeAny([_]Layer.LayerType{
  //   Layer.mergeAny(.{
  //     [_]Layer.LayerType{
  //       Layer.getConvolver(4, 4, 4, 4, Activation.getPReLU(0.25)),
  //     },
  //     [_]Layer.LayerType{
  //       Layer.getConvolver(4, 4, 4, 4, Activation.getPReLU(0.25)),
  //     },
  //   }),
  // }),
  Layer.getFlattener(),
  // Layer.getDense(28*28, Activation.getPReLU(0.5)),
  // Layer.getDense(28*28, Activation.getPReLU(0.125)),
  // Layer.getDense(14*14, Activation.getPReLU(0.125)),
  // Layer.getDense(14*14, Activation.getPReLU(0.125)),
  Layer.getDense(7*7, Activation.getPReLU(0.125)),
  Layer.getDense(7*7, Activation.getPReLU(0.125)),
  Layer.getDense(10, Activation.NormalizeSquared),
});

const MNISTIterator = MNIST.GetMinstIterator(28, 28);

fn getTester(allocator: std.mem.Allocator) !cnn.Tester {
  if (try cnn.Tester.exists()) {
    try logger.writer.print("Cnn model found, Do you want to load it? (y/n) ", .{});
    try logger.buffered.flush();

    const stdin = std.io.getStdIn().reader();
    const answer = try stdin.readByte();
    if (answer != 'n') {
      return try cnn.Tester.load();
    }
  }

  var trainer: cnn.Trainer = undefined;
  trainer.reset(rng.random());

  var mnist_iterator = try MNISTIterator.init("./datasets/train-images.idx3-ubyte", "./datasets/train-labels.idx1-ubyte", allocator);
  defer mnist_iterator.free(allocator);

  inline for (0..7) |i| {
    trainer.train(mnist_iterator.randomIterator(rng.random(), @intCast(mnist_iterator.count*(i+1))), .{
      .batch_size = @intCast(mnist_iterator.count/(10*(i+1)*(i+1))),
      .learning_rate = 100 / @exp(@as(f64, @floatFromInt(i+1))),
    }, true);
  }

  const tester = trainer.toTester();
  try tester.save();
  return tester;
}

var rng = std.Random.DefaultPrng.init(1);
var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};

pub fn main() !void {
  const allocator = gpa.allocator();
  const tester = try getTester(allocator);

  logger.writer.print("Testing...\n", .{}) catch {};
  logger.buffered.flush() catch {};

  var mnist_test_iterator = try MNISTIterator.init("./datasets/t10k-images.idx3-ubyte", "./datasets/t10k-labels.idx1-ubyte", allocator);
  defer mnist_test_iterator.free(allocator);

  const accuracy = tester.@"test"(mnist_test_iterator, false);
  logger.writer.print(">>Final Accuracy: {d:.3}%\n", .{accuracy*100}) catch {};
  logger.buffered.flush() catch {};
}

