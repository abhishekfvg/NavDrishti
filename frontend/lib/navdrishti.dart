import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

late List<CameraDescription> cameras;

class NavDrishti extends StatelessWidget {
  const NavDrishti({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {

  late CameraController controller;
  int cameraIndex = 0;

  @override
  void initState() {
    super.initState();
    initializeCamera();
  }

  Future<void> initializeCamera() async {

    cameras = await availableCameras();

    controller = CameraController(
      cameras[cameraIndex],
      ResolutionPreset.high,
    );

    await controller.initialize();

    if (!mounted) return;
    setState(() {});
  }

  Future<void> switchCamera() async {

    cameraIndex = cameraIndex == 0 ? 1 : 0;

    await controller.dispose();

    controller = CameraController(
      cameras[cameraIndex],
      ResolutionPreset.high,
    );

    await controller.initialize();

    setState(() {});
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      backgroundColor: const Color(0xFF0B0B16),

      body: Column(
        children: [

          const SizedBox(height: 40),

          // TOP BAR
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  "NavDrishti",
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                Icon(
                  Icons.settings,
                  size: 28,
                  color: Colors.white,
                )
              ],
            ),
          ),

          const SizedBox(height: 15),

          // CAMERA AREA
          Expanded(
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 20),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(25),
              ),
              child: controller.value.isInitialized
                  ? ClipRRect(
                borderRadius: BorderRadius.circular(25),
                child: FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: controller.value.previewSize!.height,
                    height: controller.value.previewSize!.width,
                    child: CameraPreview(controller),
                  ),
                ),
              )
                  : const Center(
                child: CircularProgressIndicator(),
              ),
            ),
          ),

          const SizedBox(height: 20),

          // BOTTOM CONTROLS
          Padding(
            padding: const EdgeInsets.only(bottom: 30),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [

                // HISTORY BUTTON
                const CircleAvatar(
                  radius: 30,
                  backgroundColor: Colors.white,
                  child: Icon(Icons.history, color: Colors.black),
                ),

                // MICROPHONE BUTTON
                const CircleAvatar(
                  radius: 40,
                  backgroundColor: Colors.blue,
                  child: Icon(Icons.mic, size: 35),
                ),

                // FLIP CAMERA BUTTON
                GestureDetector(
                  onTap: switchCamera,
                  child: const CircleAvatar(
                    radius: 30,
                    backgroundColor: Colors.red,
                    child: Icon(Icons.flip_camera_android),
                  ),
                ),
              ],
            ),
          )
        ],
      ),
    );
  }
}