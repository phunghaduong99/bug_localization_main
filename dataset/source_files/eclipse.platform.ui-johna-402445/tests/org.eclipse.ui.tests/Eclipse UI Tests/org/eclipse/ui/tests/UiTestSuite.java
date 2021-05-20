/*******************************************************************************
 * Copyright (c) 2000, 2013 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
package org.eclipse.ui.tests;

import junit.framework.Test;
import junit.framework.TestSuite;

import org.eclipse.ui.tests.activities.ActivitiesTestSuite;
import org.eclipse.ui.tests.api.ApiTestSuite;
import org.eclipse.ui.tests.api.StartupTest;
import org.eclipse.ui.tests.datatransfer.DataTransferTestSuite;
import org.eclipse.ui.tests.decorators.DecoratorsTestSuite;
import org.eclipse.ui.tests.dialogs.UIAutomatedSuite;
import org.eclipse.ui.tests.encoding.EncodingTestSuite;
import org.eclipse.ui.tests.fieldassist.FieldAssistTestSuite;
import org.eclipse.ui.tests.navigator.NavigatorTestSuite;
import org.eclipse.ui.tests.operations.OperationsTestSuite;
import org.eclipse.ui.tests.preferences.PreferencesTestSuite;

/**
 * Test all areas of the UI.
 */
public class UiTestSuite extends TestSuite {

	/**
	 * Returns the suite. This is required to use the JUnit Launcher.
	 */
	public static Test suite() {
		return new UiTestSuite();
	}

	/**
	 * Construct the test suite.
	 */
	public UiTestSuite() {
		// run the StartupTest first, since we need to check early that the
		// tests
		// run only after early startup has completed (bug 93518).
		addTest(new TestSuite(StartupTest.class));
		addTest(new UIAutomatedSuite());
		addTest(new ApiTestSuite());
		// addTest(new PropertySheetTestSuite());
		// addTest(new QuickAccessTestSuite());
		// addTest(new InternalTestSuite());
		addTest(new NavigatorTestSuite());
		addTest(new DecoratorsTestSuite());
		// addTest(new AdaptableTestSuite());
		// addTest(new ZoomTestSuite());
		addTest(new DataTransferTestSuite());
		addTest(new PreferencesTestSuite());
		// addTest(new DynamicPluginsTestSuite());
		// addTest(new KeysTestSuite());
		// addTest(new MultiPageEditorTestSuite());
		addTest(new ActivitiesTestSuite());
		// addTest(new CommandsTestSuite());
		// addTest(new ContextsTestSuite());
		// addTest(new DragTestSuite());
		// addTest(new ThemesTestSuite());
		// addTest(new IntroTestSuite());
		// addTest(new MenusTestSuite());
		addTest(new EncodingTestSuite());
		// addTest(new PresentationsTestSuite());
		// addTest(new TestSuite(LeakTests.class));
		// addTest(new ConcurrencyTestSuite());
		addTest(new OperationsTestSuite());
		addTest(new FieldAssistTestSuite());
		// addTest(new MultiEditorTestSuite());
		// addTest(new TestSuite(FilteredTreeTests.class));
		// addTest(new ServicesTestSuite());
		// addTest(new StatusHandlingTestSuite());
		// addTest(OpenSystemInPlaceEditorTest.suite());
		// addTest(new ProgressTestSuite());
	}
}
