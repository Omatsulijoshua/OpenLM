Pod::Spec.new do |s|
  s.name             = 'llama_flutter_android'
  s.version          = '0.3.0'
  s.summary          = 'Run GGUF models on Android, iOS, macOS, Windows, and Linux with llama.cpp (Metal + Vulkan + RPC Cluster)'
  s.description      = <<-DESC
    Flutter plugin to run GGUF models locally using llama.cpp.
    macOS implementation uses Metal for GPU acceleration and ggml-rpc for multi-device clustering.
  DESC
  s.homepage         = 'https://github.com/dragneel2074/Llama-Flutter'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Write4Me' => 'https://github.com/dragneel2074' }

  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.public_header_files = 'Classes/LlamaIosWrapper.h', 'Classes/LlamaClusterWorker.h', 'Classes/ObjcExceptionBridge.h'

  s.platform = :osx, '11.0'
  s.swift_version = '5.0'

  # Automatically build llama.xcframework before compiling if not yet built
  s.prepare_command = <<-CMD
    bash ../scripts/build_llama_macos.sh
  CMD

  s.vendored_frameworks = 'Frameworks/llama.xcframework'

  s.frameworks = 'Foundation', 'Metal', 'MetalKit', 'Accelerate', 'IOKit'
  s.libraries = 'c++'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'OTHER_CPLUSPLUSFLAGS' => '-fexceptions -frtti -O3',
    'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/Frameworks/llama.xcframework/macos-arm64_x86_64/Headers"',
  }

  s.dependency 'FlutterMacOS'
end
