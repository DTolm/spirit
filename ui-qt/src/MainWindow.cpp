#include <QtWidgets>

#include "MainWindow.hpp"
#include "PlotWidget.hpp"

#include "Interface_System.h"
#include "Interface_Chain.h"
#include "Interface_Collection.h"
#include "Interface_Simulation.h"
#include "Interface_Configurations.h"
#include "Interface_Log.h"


// TODO: Replace these
#include "IO.hpp"
#include "State.hpp"
/////


MainWindow::MainWindow(std::shared_ptr<State> state)
{
	// State
	this->state = state;
	// Widgets
	this->spinWidget = new SpinWidget(this->state);
	this->controlWidget = new ControlWidget(this->state, this->spinWidget);
	this->settingsWidget = new SettingsWidget(this->state, this->spinWidget);
	this->plotsWidget = new PlotsWidget(this->state);
	this->debugWidget = new DebugWidget(this->state);

	//this->setFocus(Qt::StrongFocus);
	this->setFocusPolicy(Qt::StrongFocus);

	// Fix text size on OSX
    #ifdef Q_OS_MAC
        this->setStyleSheet("QWidget{font-size:10pt}");
    #else
        this->setStyleSheet("QWidget{font-size:8pt}");
    #endif
    
	// Setup User Interface
    this->setupUi(this);

	// DockWidgets: tabify for Plots and Debug
	this->tabifyDockWidget(this->dockWidget_Plots, this->dockWidget_Debug);
	this->dockWidget_Plots->raise();
	this->dockWidget_Debug->hide();
	// DockWidgets: assign widgets
	this->dockWidget_Settings->setWidget(this->settingsWidget);
	this->dockWidget_Plots->setWidget(this->plotsWidget);
	this->dockWidget_Debug->setWidget(this->debugWidget);

	// Add Widgets to UIs grids
	this->gridLayout->addWidget(this->spinWidget, 0, 0, 1, 1);
	this->gridLayout_2->addWidget(this->controlWidget, 0, 0, 1, 1);

	// Read Window settings of last session
	readSettings();

	
	// Set up Update Timers
	m_timer = new QTimer(this);
	//m_timer_plots = new QTimer(this);
	//m_timer_spins = new QTimer(this);


	// File Menu
	connect(this->actionLoad_Configuration, SIGNAL(triggered()), this, SLOT(load_Configuration()));
	connect(this->actionLoad_Spin_Configuration, SIGNAL(triggered()), this, SLOT(load_Spin_Configuration()));
	connect(this->actionLoad_SpinChain_Configuration, SIGNAL(triggered()), this, SLOT(load_SpinChain_Configuration()));
	connect(this->actionSave_Energies, SIGNAL(triggered()), this, SLOT(save_Energies()));
	connect(this->action_Save_Spin_Configuration, SIGNAL(triggered()), SLOT(save_Spin_Configuration()));
	connect(this->actionSave_SpinChain_Configuration, SIGNAL(triggered()), this, SLOT(save_SpinChain_Configuration()));

	// View Menu
	connect(this->actionShow_Settings, SIGNAL(triggered()), this, SLOT(view_toggleSettings()));
	connect(this->actionShow_Plots, SIGNAL(triggered()), this, SLOT(view_togglePlots()));
	connect(this->actionShow_Debug, SIGNAL(triggered()), this, SLOT(view_toggleDebug()));
	connect(this->actionToggle_large_visualisation, SIGNAL(triggered()), this, SLOT(view_toggle_fullscreen_spins()));

	// Help Menu
	connect(this->actionKey_Bindings, SIGNAL(triggered()), this, SLOT(keyBindings()));	
	connect(this->actionAbout_this_Application, SIGNAL(triggered()), this, SLOT(about()));

	// Status Bar
	//		FPS
	this->m_Label_FPS = new QLabel;
	this->m_Label_FPS->setText("FPS: 0");
	Ui::MainWindow::statusBar->addPermanentWidget(m_Label_FPS);
	//		NOS
	this->m_Label_NOS = new QLabel;
	this->m_Label_NOS->setText("NOS: 0");
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOS);
	//		NOI
	this->m_Label_NOI = new QLabel;
	this->m_Label_NOI->setText("NOI: 0");
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOI);
	//		NOC
	this->m_Label_NOC = new QLabel;
	this->m_Label_NOC->setText("NOC: 0");
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOC);
	//		Initialisations & connect
	this->createStatusBar();
	connect(m_timer, &QTimer::timeout, this, &MainWindow::updateStatusBar);

	// Plots Widget
	//connect(m_timer_plots, &QTimer::timeout, this->plotsWidget->energyPlot, &PlotWidget::update);	// this currently resets the user's interaction (movement, zoom)

	// Spins Widget
	//connect(m_timer_spins, &QTimer::timeout, this->spinWidget, &Spin_Widget::update);
	

	// Event Filter
	//this->installEventFilter(this);

	// Start Timers
	m_timer->start(200);
	//m_timer_plots->start(100);
	//m_timer_spins->start(100);
	//m_timer_debug->start(100);



	Ui::MainWindow::statusBar->showMessage(tr("Ready"), 5000);
}


void MainWindow::view_toggle_fullscreen_spins()
{
	if (this->fullscreen_spins)
	{
		this->fullscreen_spins = false;
		this->controlWidget->show();
	}
	else
	{
		this->fullscreen_spins = true;
		this->controlWidget->hide();
	}
}


void MainWindow::keyPressEvent(QKeyEvent *k)
{
	// Key Sequences
	if (k->matches(QKeySequence::Copy))
	{
		// Copy a Spin System
		Chain_Image_to_Clipboard(state.get());
	}
	else if (k->matches(QKeySequence::Cut))
	{
		if (Chain_Get_NOI(state.get()) > 1)
		{
			this->controlWidget->stop_current();

			// Cut a Spin System
			Chain_Image_to_Clipboard(state.get());

			int idx = System_Get_Index(state.get());
			if (idx > 0) Chain_prev_Image(this->state.get());
			this->spinWidget->update();			
			//else this->nextImagePressed();

			if (Chain_Delete_Image(state.get(), idx)) 
			{
				// Make the llg_threads vector smaller
				this->threads_llg.erase(threads_llg.begin() + idx);
			}
		}
	}
	else if (k->matches(QKeySequence::Paste))
	{
		// Paste a Spin System
		this->controlWidget->stop_current();
		Chain_Replace_Image(state.get());
		// Update the chain's data (primarily for the plot)
		Chain_Update_Data(state.get());
		// Update Visualisation
		this->spinWidget->update();
	}

	// Custom Key Sequences
	else if (k->modifiers() & Qt::ControlModifier)
	{
		int idx = System_Get_Index(state.get());
		switch (k->key())
		{
			// CTRL+Left - Paste image to left of current image
			case Qt::Key_Left:
				// Insert Image
				Chain_Insert_Image_Before(state.get());
				// Update the chain's data (primarily for the plot)
				Chain_Update_Data(state.get());
				// Make the llg_threads vector larger
				this->threads_llg.insert(threads_llg.begin()+idx, std::thread());
				break;

			// CTRL+Right - Paste image to right of current image
			case Qt::Key_Right:
				// Insert Image
				Chain_Insert_Image_After(state.get());
				// Update the chain's data (primarily for the plot)
				Chain_Update_Data(state.get());
				// Make the llg_threads vector larger
				this->threads_llg.insert(threads_llg.begin()+idx+1, std::thread());
				// Switch to the inserted image
				Chain_next_Image(this->state.get());
				this->spinWidget->update();
				break;
			
			// CTRL+F - Fullscreen mode
			case Qt::Key_F:
				this->view_toggle_fullscreen_spins();
				break;
		}
	}
	
	// Single Keys
	else
	switch (k->key())
	{
		// Escape: try to return focus to MainWindow
		case Qt::Key_Escape:
			this->setFocus();
			break;
		// Up: ...
		case Qt::Key_Up:
			break;
		// Left: switch to image left of current image
		case Qt::Key_Left:
			Chain_prev_Image(this->state.get());
			this->spinWidget->update();
			break;
		// Left: switch to image left of current image
		case Qt::Key_Right:
			Chain_next_Image(this->state.get());
			this->spinWidget->update();
			break;
		// Down: ...
		case Qt::Key_Down:
			break;
		// Space: Play and Pause
		case Qt::Key_Space:
			this->controlWidget->play_pause();
			break;
		// F1: Show key bindings
		case Qt::Key_F1:
			this->keyBindings();
			break;
		// F2: Toggle settings widget
		case Qt::Key_F2:
			this->view_toggleSettings();
			break;
		// F3: Toggle Plots widget
		case Qt::Key_F3:
			this->view_togglePlots();
			break;
		// F2: Toggle debug widget
		case Qt::Key_F4:
			this->view_toggleDebug();
			break;
		// 0: ...
		case Qt::Key_0:
			break;
		// 1: Select tab 1 of settings widget
		case Qt::Key_1:
			this->settingsWidget->SelectTab(0);
			break;
		// 2: Select tab 2 of settings widget 
		case Qt::Key_2:
			this->settingsWidget->SelectTab(1);
			break;
		// 3: Select tab 3 of settings widget
		case Qt::Key_3:
			this->settingsWidget->SelectTab(2);
			break;
		// 4: Select tab 4 of settings widget
		case Qt::Key_4:
			this->settingsWidget->SelectTab(3);
			break;
		// 5: Select tab 5 of settings widget
		case Qt::Key_5:
			this->settingsWidget->SelectTab(4);
			break;
		// Delete: Delete current image
		case Qt::Key_Delete:
			if (Chain_Get_NOI(state.get()) > 1)
			{
				this->controlWidget->stop_current();

				int idx = System_Get_Index(state.get());
				if (idx > 0) Chain_prev_Image(this->state.get());
				//else this->nextImagePressed();
				if (Chain_Delete_Image(state.get(), idx)) 
				{
					// Make the llg_threads vector smaller
					this->threads_llg.erase(threads_llg.begin() + idx);
				}

				Log_Send(state.get(), Log_Level_Info, Log_Sender_UI, ("Deleted image " + std::to_string(System_Get_Index(state.get()))).c_str());
			}
			this->spinWidget->update();
			break;
	}
}


void MainWindow::view_toggleDebug()
{
	if (this->dockWidget_Debug->isVisible()) this->dockWidget_Debug->hide();
	else this->dockWidget_Debug->show();
}

void MainWindow::view_togglePlots()
{
	if (this->dockWidget_Plots->isVisible()) this->dockWidget_Plots->hide();
	else this->dockWidget_Plots->show();
}

void MainWindow::view_toggleSettings()
{
	if (this->dockWidget_Settings->isVisible()) this->dockWidget_Settings->hide();
	else this->dockWidget_Settings->show();
}


void MainWindow::createStatusBar()
{
	// Remove previous IPS labels
	for (unsigned int i = 0; i < this->m_Labels_IPS.size(); ++i)
	{
		Ui::MainWindow::statusBar->removeWidget(this->m_Labels_IPS[i]);
	}

	// Get Methods' IPS
	auto v_ips = Simulation_Get_IterationsPerSecond(state.get());

	// Create IPS Labels and add them to the statusBar
	this->m_Labels_IPS = std::vector<QLabel*>();
	for (unsigned int i = 0; i < v_ips.size(); ++i)
	{
		this->m_Labels_IPS.push_back(new QLabel);
		this->m_Labels_IPS[i]->setText("IPS [-]: -");
		Ui::MainWindow::statusBar->addPermanentWidget(m_Labels_IPS[i]);
	}

	//		FPS
	Ui::MainWindow::statusBar->removeWidget(this->m_Label_FPS);
	this->m_Label_FPS = new QLabel;
	this->m_Label_FPS->setText("FPS: -");
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_FPS);

	//		NOS
	Ui::MainWindow::statusBar->removeWidget(this->m_Label_NOS);
	this->m_Label_NOS = new QLabel;
	this->m_Label_NOS->setText(QString::fromLatin1("NOS: ") + QString::number(System_Get_NOS(this->state.get())));
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOS);

	//		NOI
	Ui::MainWindow::statusBar->removeWidget(this->m_Label_NOI);
	this->m_Label_NOI = new QLabel;
	this->m_Label_NOI->setText(QString::fromLatin1("NOI: ") + QString::number(Chain_Get_NOI(this->state.get())));
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOI);

	//		NOC
	Ui::MainWindow::statusBar->removeWidget(this->m_Label_NOC);
	this->m_Label_NOC = new QLabel;
	this->m_Label_NOC->setText(QString::fromLatin1("NOC: ") + QString::number(Collection_Get_NOC(this->state.get())));
	Ui::MainWindow::statusBar->addPermanentWidget(this->m_Label_NOC);
}


void MainWindow::updateStatusBar()
{
	this->m_Label_FPS->setText(QString::fromLatin1("FPS: ") + QString::number((int)this->spinWidget->getFramesPerSecond()));
	auto v_ips = Simulation_Get_IterationsPerSecond(state.get());
	for (unsigned int i = 0; i < m_Labels_IPS.size() && i < v_ips.size(); ++i)
	{
		this->m_Labels_IPS[i]->setText(QString::fromLatin1("IPS [") + QString::number(i+1) + QString::fromLatin1("]: ") + QString::number((int)v_ips[i]));
	}
}


void MainWindow::about()
{
	QMessageBox::about(this, tr("About Spirit"),
		QString::fromLatin1("The <b>Spirit</b> application incorporates intuitive visualisation,<br>"
			"powerful <b>Spin Dynamics</b> and <b>Nudged Elastic Band</b> tools<br>"
			"into a cross-platform user interface.<br>"
			"<br>"
			"Libraries used are<br>"
			"  - VTK 7<br>"
			"  - QT 5.5<br>"
			"<br>"
			"This has been developed by<br>"
			"  - Gideon M�ller (<a href=\"mailto:g.mueller@fz-juelich.de\">g.mueller@fz-juelich.de</a>)<br>"
			"  - Daniel Sch�rhoff (<a href=\"mailto:d.schuerhoff@fz-juelich.de\">d.schuerhoff@fz-juelich.de</a>)<br>"
			"at the Institute for Advanced Simulation 1 of the Forschungszentrum J�lich.<br>"
			"For more information about us, visit the <a href=\"http://www.fz-juelich.de/pgi/pgi-1/DE/Home/home_node.html\">IAS-1 Website</a><br>"
			"<br>"
			"<b>Copyright 2016</b><br>"));
}

void MainWindow::keyBindings()
{
	QMessageBox::about(this, tr("Spirit UI Key Bindings"),
		QString::fromLatin1("The <b>Key Bindings</b> are as follows:<br>"
			"<br>"
			" - <b>F1</b>:      Show this<br>"
			" - <b>F2</b>:      Toggle Settings<br>"
			" - <b>F3</b>:      Toggle Plots<br>"
			" - <b>F4</b>:      Toggle Debug<br>"
			" - <b>Ctrl+F</b>:  Toggle large visualisation<br>"
			"<br>"
			" - <b>1-5</b>:     Select Tab in Settings<br>"
			"<br>"
			" - <b>Arrows</b>:  Switch between arrows and chains<br>"
			" - <b>WASD</b>:    Move the camera around (not yet functional)<br>"
			" - <b>Space</b>:   Play/Pause<br>"
			" - <b>Escape</b>:  Try to return focus to main UI (does not always work)<br>"
			"<br>"
			" - <b>Ctrl+X</b>:           Cut   Image<br>"
			" - <b>Ctrl+C</b>:           Copy  Image<br>"
			" - <b>Ctrl+V</b>:           Paste Image at current index<br>"
			" - <b>Ctrl+Left/Right</b>:  Insert left/right of current index<br>"
			" - <b>Del</b>:              Delete image<br>"));
}

void MainWindow::return_focus()
{
	auto childWidgets = this->findChildren<QWidget *>();
	for (int i = 0; i <childWidgets.count(); ++i)
	{
		childWidgets.at(i)->clearFocus();
	}
}



void MainWindow::save_Spin_Configuration()
{
	auto fileName = QFileDialog::getSaveFileName(this, tr("Save Spin Configuration"), "./output", tr("Spin Configuration (*.txt)"));
	if (!fileName.isEmpty()) {
		Utility::IO::Append_Spin_Configuration(this->state->active_image, 0, string_q2std(fileName));
	}
}

void MainWindow::load_Spin_Configuration()
{
	auto fileName = QFileDialog::getOpenFileName(this, tr("Load Spin Configuration"), "./input", tr("Spin Configuration (*.txt *.csv)"));
	if (!fileName.isEmpty()) {
		QFileInfo fi(fileName);
		auto qs_type = fi.completeSuffix();
		Utility::IO::VectorFileFormat type;
		if (qs_type == "csv") type = Utility::IO::VectorFileFormat::CSV_POS_SPIN;
		else type = Utility::IO::VectorFileFormat::WHITESPACE_SPIN;
		Utility::IO::Read_Spin_Configuration(this->state->active_image, string_q2std(fileName), type);
	}
	this->spinWidget->update();
}

void MainWindow::save_SpinChain_Configuration()
{
	auto fileName = QFileDialog::getSaveFileName(this, tr("Save SpinChain Configuration"), "./output", tr("Spin Configuration (*.txt)"));
	if (!fileName.isEmpty()) {
		Utility::IO::Save_SpinChain_Configuration(this->state->active_chain, string_q2std(fileName));
	}
}

void MainWindow::load_SpinChain_Configuration()
{
	auto fileName = QFileDialog::getOpenFileName(this, tr("Load Spin Configuration"), "./input", tr("Spin Configuration (*.txt)"));
	if (!fileName.isEmpty()) {
		Utility::IO::Read_SpinChain_Configuration(this->state->active_chain, string_q2std(fileName));
	}
	this->spinWidget->update();
}


void MainWindow::load_Configuration()
{
	int idx_img = System_Get_Index(state.get());
	// Read Spin System from cfg
	auto fileName = QFileDialog::getOpenFileName(this, tr("Open Config"), "./input", tr("Config (*.cfg)"));
	if (!fileName.isEmpty())
	{
		// TODO: use interface_system function
		std::shared_ptr<Data::Spin_System> sys = Utility::IO::Spin_System_from_Config(string_q2std(fileName));
		// Filter for unacceptable differences to other systems in the chain
		bool acceptable = true;
		for (int i = 0; i < Chain_Get_NOI(state.get()); ++i)
		{
			if (state->active_chain->images[i]->nos != sys->nos) acceptable = false;
			// Currently the SettingsWidget does not support different images being isotropic AND anisotropic at the same time
			if (state->active_chain->images[i]->hamiltonian->Name() != sys->hamiltonian->Name()) acceptable = false;
		}
		// Set current image
		if (acceptable)
		{
			this->state->active_chain->images[idx_img] = sys;
			Configuration_Random(state.get());
		}
		else QMessageBox::about(this, tr("About Spirit"),
			tr("The resulting Spin System would have different NOS\n"
				"or isotropy status than one or more of the other\n"
				"images in the chain!\n"
				"\n"
				"The system has thus not been reset!"));
	}
}

void MainWindow::save_Energies()
{
	this->return_focus();
	auto fileName = QFileDialog::getSaveFileName(this, tr("Save Energies"), "./output", tr("Text (*.txt)"));
	if (!fileName.isEmpty()) {
		Utility::IO::Save_Energies(*state->active_chain, 0, string_q2std(fileName));
	}
}

void MainWindow::readSettings()
{
    QSettings settings("Spirit Code", "Spirit");
    restoreGeometry(settings.value("geometry").toByteArray());
    restoreState(settings.value("windowState").toByteArray());
	bool fullscreen = settings.value("fullscreenSpins").toBool();
	if (fullscreen) this->view_toggle_fullscreen_spins();

	// Settings Dock
	settings.beginGroup("SettingsDock");
	// dockWidget_Settings->setFloating(settings.value("docked").toBool());
	// addDockWidget((Qt::DockWidgetArea)settings.value("dockarea", Qt::RightDockWidgetArea).toInt(), dockWidget_Settings);
	// dockWidget_Settings->setHidden(settings.value("hidden").toBool());
	dockWidget_Settings->resize(settings.value("size", QSize(1, 1)).toSize());
	dockWidget_Settings->move(settings.value("pos", QPoint(200, 200)).toPoint());
	settings.endGroup();

	// Plots Dock
	settings.beginGroup("PlotsDock");
	// dockWidget_Plots->setFloating(settings.value("docked").toBool());
	// addDockWidget((Qt::DockWidgetArea)settings.value("dockarea", Qt::RightDockWidgetArea).toInt(), dockWidget_Plots);
	// dockWidget_Plots->setHidden(settings.value("hidden").toBool());
	dockWidget_Plots->resize(settings.value("size", QSize(1, 1)).toSize());
	dockWidget_Plots->move(settings.value("pos", QPoint(200, 200)).toPoint());
	settings.endGroup();

	// Debug Dock
	settings.beginGroup("DebugDock");
	// dockWidget_Debug->setFloating(settings.value("docked").toBool());
	// addDockWidget((Qt::DockWidgetArea)settings.value("dockarea", Qt::RightDockWidgetArea).toInt(), dockWidget_Debug);
	// dockWidget_Debug->setHidden(settings.value("hidden").toBool());
	dockWidget_Debug->resize(settings.value("size", QSize(1, 1)).toSize());
	dockWidget_Debug->move(settings.value("pos", QPoint(200, 200)).toPoint());
	settings.endGroup();
}

void MainWindow::writeSettings()
{
	QSettings settings("Spirit Code", "Spirit");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("windowState", saveState());
	settings.setValue("fullscreenSpins", this->fullscreen_spins);
	
	// Settings Dock
	settings.beginGroup("SettingsDock");
	// settings.setValue("dockarea", dockWidgetArea(dockWidget_Settings));
	// settings.setValue("docked", dockWidget_Settings->isFloating());
	// settings.setValue("hidden", dockWidget_Settings->isHidden());
	settings.setValue("size", dockWidget_Settings->size());
	settings.setValue("pos", dockWidget_Settings->pos());
	settings.endGroup();

	// Plots Dock
	settings.beginGroup("PlotsDock");
	// settings.setValue("dockarea", dockWidgetArea(dockWidget_Plots));
	// settings.setValue("docked", dockWidget_Plots->isFloating());
	// settings.setValue("hidden", dockWidget_Plots->isHidden());
	settings.setValue("size", dockWidget_Plots->size());
	settings.setValue("pos", dockWidget_Plots->pos());
	settings.endGroup();

	// Debug Dock
	settings.beginGroup("DebugDock");
	// settings.setValue("dockarea", dockWidgetArea(dockWidget_Debug));
	// settings.setValue("docked", dockWidget_Debug->isFloating());
	// settings.setValue("hidden", dockWidget_Debug->isHidden());
	settings.setValue("size", dockWidget_Debug->size());
	settings.setValue("pos", dockWidget_Debug->pos());
	settings.endGroup();
}


void MainWindow::closeEvent(QCloseEvent *event)
{
	this->controlWidget->stop_all();
	
	writeSettings();
	event->accept();
}