package maosKernel;

import sun.misc.*;


public class TimeoutSignalHandler implements SignalHandler {
	public void handle(Signal arg){
		System.out.println("Unsuccessful");
		System.exit(0);
	}
}

